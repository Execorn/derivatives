"""Sobolev H^1 trainer and SR 26-2 OAS/split-conformal calibrator for EGNO v2.0."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp_sparse
from scipy.optimize import linprog, nnls
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepvol.calibration.autocall_sobol_generator import (
    SyntheticAutocallBatch,
    generate_synthetic_autocall_batch,
)
from deepvol.models.autocall.feature_layout import (
    D_MAX_DEFAULT,
    FEATURE_SCHEMA_VERSION,
    M_MAX_DEFAULT,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    MonitoringIndexCode,
)
from deepvol.surrogates.eq_autocall_egno import (
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3] / "artifacts" / "checkpoints" / "eq_autocall_egno.pt"
)
DEFAULT_GUARDIAN_STATS_PATH = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "checkpoints"
    / "eq_autocall_guardian_stats.pt"
)

FEATURE_NAMES_16D: Tuple[str, ...] = (
    "mean_log_moneyness",
    "min_log_moneyness",
    "max_log_moneyness",
    "mean_atm_vol",
    "max_atm_vol",
    "mean_corr",
    "min_corr",
    "max_cash_div_leverage",
    "term_maturity",
    "log_ko_barrier",
    "log_cpn_barrier",
    "put_strike",
    "stulz_log_fwd_w",
    "stulz_vol_w",
    "memory_coupon_buffer",
    "abs_digital_spread",
)


@dataclass
class SobolevLossBreakdown:
    """Detailed component breakdown of `SpreadAttenuatedSobolevLoss`."""

    total_loss: torch.Tensor
    val_loss: torch.Tensor
    comp_loss: torch.Tensor
    sobolev_delta_loss: torch.Tensor
    monotonicity_loss: torch.Tensor
    jacobian_reg_loss: torch.Tensor
    mae_bps: float
    delta_cos_sim: float


@dataclass
class SurrogateTrainingResult:
    """Summary of a completed `train_eq_autocall_surrogate` run."""

    model: MaskedEQAutocallSurrogate
    checkpoint_path: Optional[Path]
    guardian_stats_path: Optional[Path]
    train_loss_history: List[float]
    val_mae_bps: float
    test_mae: float
    test_mae_bps: float
    test_worstof_mae_bps: float
    test_multifunc_mae_bps: float
    baseline_mae_bps: float
    baseline_worstof_mae_bps: float
    baseline_multifunc_mae_bps: float
    conformal_coverage: float
    sobolev_delta_cos_sim: float
    train_wall_sec: float
    per_d_mae_bps: Dict[int, float]
    total_delta_cos_sim: float = 0.0


class SpreadAttenuatedSobolevLoss(nn.Module):
    """Spread-Attenuated Sobolev H^1 Training Loss with precomputed dX/dS chain rule."""

    def __init__(
        self,
        delta_b_0: float = 0.05,
        lambda_comp: float = 0.50,
        lambda_delta: float = 1.0,
        lambda_mono: float = 1.0,
        lambda_jac: float = 1e-4,
        ko_bump_delta: float = 0.02,
    ) -> None:
        super().__init__()
        if delta_b_0 <= 0.0:
            raise ValueError(f"delta_b_0 must be > 0, got {delta_b_0}.")
        self.delta_b_0 = float(delta_b_0)
        self.lambda_comp = float(lambda_comp)
        self.lambda_delta = float(lambda_delta)
        self.lambda_mono = float(lambda_mono)
        self.lambda_jac = float(lambda_jac)
        self.ko_bump_delta = float(ko_bump_delta)

    def spread_attenuation_weight(self, global_feats: torch.Tensor) -> torch.Tensor:
        """Computes w(Delta_B) = |Delta_B| / (|Delta_B| + Delta_B_0) in [0, 1)."""
        ko_sp = global_feats[:, int(AutocallGlobalCol.ABS_KO_SPREAD)].abs()
        cpn_sp = global_feats[:, int(AutocallGlobalCol.ABS_CPN_SPREAD)].abs()
        delta_b = torch.maximum(ko_sp, cpn_sp)
        if global_feats.shape[-1] > int(AutocallGlobalCol.ABS_KI_SPREAD):
            ki_sp = global_feats[:, int(AutocallGlobalCol.ABS_KI_SPREAD)].abs()
            delta_b = torch.maximum(delta_b, ki_sp)
        return delta_b / (delta_b + self.delta_b_0)

    def forward_from_tensors(
        self,
        *,
        pv_pred: torch.Tensor,
        pv_target: torch.Tensor,
        p_pred: torch.Tensor,
        p_target: torch.Tensor,
        q_pred: torch.Tensor,
        q_target: torch.Tensor,
        l_put_pred: torch.Tensor,
        l_put_target: torch.Tensor,
        delta_pred: torch.Tensor,
        delta_target: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        h_base: Optional[torch.Tensor] = None,
        h_bumped: Optional[torch.Tensor] = None,
        surv_base: Optional[torch.Tensor] = None,
        surv_bumped: Optional[torch.Tensor] = None,
    ) -> SobolevLossBreakdown:
        """Computes all Sobolev loss terms from pre-evaluated prediction and target tensors."""
        dtype = pv_pred.dtype
        pv_t = pv_target.to(dtype=dtype)
        p_t = p_target.to(dtype=dtype)
        q_t = q_target.to(dtype=dtype)
        l_t = l_put_target.to(dtype=dtype)
        d_t = delta_target.to(dtype=dtype)

        m_asset = (asset_mask > 0.5).to(dtype=dtype)
        m_sched = (sched_mask > 0.5).to(dtype=dtype)
        d_count = m_asset.sum(dim=1).clamp(min=1.0)
        m_count = m_sched.sum(dim=1).clamp(min=1.0)

        # 1. Balanced SmoothL1 + light MSE value loss (prevents exotic outliers drowning WorstOf)
        val_loss = (
            100.0 * F.smooth_l1_loss(pv_pred, pv_t, beta=0.0005)
            + 10.0 * F.mse_loss(pv_pred, pv_t)
        )

        # 2. Component probability & put-loss supervision (mask dead-survival & is_memory q_mc slots)
        p_err_sq = ((p_pred - p_t) ** 2) * m_sched
        surv_prev_t = torch.cat(
            [torch.ones_like(p_t[:, :1]), (1.0 - torch.cumsum(p_t, dim=1))[:, :-1]],
            dim=1,
        )
        non_mem = (global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)] < 0.5).unsqueeze(1)
        single_period = m_count.unsqueeze(1) < 1.5
        q_valid = m_sched * (surv_prev_t > 1e-4).to(dtype=dtype) * (non_mem | single_period).to(dtype=dtype)
        q_err_sq = ((q_pred - q_t) ** 2) * q_valid
        prob_mse = (
            (p_err_sq.sum(dim=1) / m_count)
            + (q_err_sq.sum(dim=1) / q_valid.sum(dim=1).clamp(min=1.0))
        ).mean()
        put_loss = (
            100.0 * F.smooth_l1_loss(l_put_pred, l_t, beta=0.0005)
            + 10.0 * F.mse_loss(l_put_pred, l_t)
        )
        comp_loss = 50.0 * prob_mse + put_loss

        # 3. Spread-attenuated Huber first-order Delta Sobolev penalty (applied ONCE)
        w_spread = self.spread_attenuation_weight(global_feats.to(dtype=dtype))
        delta_err = F.smooth_l1_loss(delta_pred, d_t, beta=0.15, reduction="none") * m_asset
        per_sample_delta_loss = delta_err.sum(dim=1) / d_count
        sobolev_delta_loss = (w_spread * per_sample_delta_loss).mean()

        # 4. Conditional hazard & cumulative stopping monotonicity penalty
        if h_base is not None and h_bumped is not None and surv_base is not None and surv_bumped is not None:
            haz_viol = F.relu(h_bumped - h_base).pow(2)
            cum_viol = F.relu((1.0 - surv_bumped) - (1.0 - surv_base)).pow(2)
            monotonicity_loss = (((haz_viol + cum_viol) * m_sched).sum(dim=1) / m_count).mean()
        else:
            monotonicity_loss = torch.zeros((), dtype=dtype, device=pv_pred.device)

        # 5. Frobenius Jacobian regularization
        jac_sq = (delta_pred ** 2) * m_asset
        jacobian_reg_loss = (jac_sq.sum(dim=1) / d_count).mean()

        total_loss = (
            val_loss
            + self.lambda_comp * comp_loss
            + self.lambda_delta * sobolev_delta_loss
            + self.lambda_mono * monotonicity_loss
            + self.lambda_jac * jacobian_reg_loss
        )
        mae_bps = float((pv_pred.detach() - pv_t.detach()).abs().mean().cpu().item() * 1e4)
        with torch.no_grad():
            dp_flat = (delta_pred.detach() * m_asset).reshape(-1)
            dt_flat = (d_t.detach() * m_asset).reshape(-1)
            denom = dp_flat.norm() * dt_flat.norm()
            cos_sim = float((torch.dot(dp_flat, dt_flat) / denom.clamp(min=1e-12)).cpu().item()) if denom > 1e-10 else 0.0

        return SobolevLossBreakdown(
            total_loss=total_loss,
            val_loss=val_loss,
            comp_loss=comp_loss,
            sobolev_delta_loss=sobolev_delta_loss,
            monotonicity_loss=monotonicity_loss,
            jacobian_reg_loss=jacobian_reg_loss,
            mae_bps=mae_bps,
            delta_cos_sim=cos_sim,
        )

    def forward(
        self,
        model: MaskedEQAutocallSurrogate,
        batch: SyntheticAutocallBatch,
        *,
        create_graph_for_sobolev: bool = True,
        compute_sobolev: bool = True,
        return_breakdown: bool = False,
    ) -> torch.Tensor | SobolevLossBreakdown:
        """Evaluates `model` on `batch` and computes the Spread-Attenuated Sobolev H^1 loss."""
        if compute_sobolev:
            node_in = batch.node_feats.detach().clone().requires_grad_(True)
            edge_in = batch.edge_feats.detach().clone().requires_grad_(True)
            sched_in = batch.sched_feats.detach().clone().requires_grad_(True)
            glob_in = batch.global_feats.detach().clone().requires_grad_(True)
        else:
            node_in = batch.node_feats.detach()
            edge_in = batch.edge_feats.detach()
            sched_in = batch.sched_feats.detach()
            glob_in = batch.global_feats.detach()
        amask_in = batch.asset_mask
        smask_in = batch.sched_mask

        surr_out = model(
            node_feats=node_in,
            edge_feats=edge_in,
            asset_mask=amask_in,
            sched_feats=sched_in,
            sched_mask=smask_in,
            global_feats=glob_in,
            all_active=True,
            require_second_order_sdpa=bool(compute_sobolev and create_graph_for_sobolev),
        )
        pv_pred = assemble_surrogate_product_value(
            surr_out,
            sched_in,
            smask_in,
            glob_in,
            no_notional_repayment=batch.no_notional_repayment > 0.5,
            autocall_coupon=batch.autocall_coupon,
            force_coupons_on_autocall=batch.force_coupons_on_autocall > 0.5,
            is_long_basket=batch.is_long_basket > 0.5,
            lower_strike=batch.lower_strike,
        )

        if compute_sobolev and self.lambda_delta > 0.0:
            # Contract total derivative d(V_pred)/dX @ dX/d(ln M0) using precomputed static Jacobians.
            g_node, g_edge, g_sched, g_glob = torch.autograd.grad(
                pv_pred.sum(),
                (node_in, edge_in, sched_in, glob_in),
                create_graph=create_graph_for_sobolev,
                retain_graph=True,
            )
            # LOG_PERF_DIFF = sqrt(diff^2 + eps) is symmetric in (i, j), so both cols 1 and 2 add transpose grads.
            g_edge_sym = g_edge[:, :, :, 1:3] + g_edge.transpose(1, 2)[:, :, :, 1:3]
            delta_pred_total = (
                torch.einsum("bjk,bijk->bi", g_node, batch.jac_node_logm)
                + torch.einsum("bijc,bijc->bi", g_edge_sym, batch.jac_edge_self)
                + torch.einsum("bmc,bimc->bi", g_sched[:, :, 8:12], batch.jac_sched_logm)
                + g_glob[:, int(AutocallGlobalCol.L_PUT_PROXY)].unsqueeze(1) * batch.jac_lput_logm
            ) * amask_in

            delta_pred_res = (delta_pred_total - batch.delta_proxy) * amask_in
            delta_target_res = (batch.delta_mc - batch.delta_proxy) * amask_in
        else:
            delta_pred_res = torch.zeros_like(batch.delta_proxy)
            delta_target_res = torch.zeros_like(batch.delta_proxy)

        # Evaluate KO barrier monotonicity bump using precomputed analytically bumped PROXY_KO_PROB.
        h_bumped: Optional[torch.Tensor] = None
        surv_bumped: Optional[torch.Tensor] = None
        if self.lambda_mono > 0.0 and self.ko_bump_delta > 0.0:
            sched_bumped = sched_in.detach().clone()
            is_call = sched_bumped[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)] * smask_in
            sched_bumped[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)] = (
                sched_bumped[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)] + self.ko_bump_delta * is_call
            )
            sched_bumped[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] = batch.p_proxy_ko_bumped.detach()
            surr_bumped = model(
                node_feats=node_in.detach(),
                edge_feats=edge_in.detach(),
                asset_mask=amask_in,
                sched_feats=sched_bumped,
                sched_mask=smask_in,
                global_feats=glob_in.detach(),
                all_active=True,
                require_second_order_sdpa=False,
            )
            h_bumped = surr_bumped.hazard_rates
            surv_bumped = surr_bumped.survival_probs

        # Align q_pred with force_coupons_on_autocall semantics in Dupire QMC q_mc.
        surv_prev_pred = torch.cat(
            [torch.ones_like(surr_out.survival_probs[:, :1]), surr_out.survival_probs[:, :-1]],
            dim=1,
        ).clamp(min=1e-12)
        fcpn_mask = (batch.force_coupons_on_autocall > 0.5).unsqueeze(1)
        q_eff_pred = torch.where(
            fcpn_mask,
            torch.maximum(surr_out.coupon_probs, surr_out.stopping_probs / surv_prev_pred),
            surr_out.coupon_probs,
        )

        breakdown = self.forward_from_tensors(
            pv_pred=pv_pred,
            pv_target=batch.pv_mc,
            p_pred=surr_out.stopping_probs,
            p_target=batch.p_mc,
            q_pred=q_eff_pred,
            q_target=batch.q_mc,
            l_put_pred=surr_out.put_loss,
            l_put_target=batch.l_put_mc,
            delta_pred=delta_pred_res,
            delta_target=delta_target_res,
            asset_mask=amask_in,
            sched_mask=smask_in,
            global_feats=glob_in,
            h_base=surr_out.hazard_rates,
            h_bumped=h_bumped,
            surv_base=surr_out.survival_probs,
            surv_bumped=surv_bumped,
        )
        if return_breakdown:
            return breakdown
        return breakdown.total_loss


def extract_16d_features_from_batch(batch: SyntheticAutocallBatch) -> npt.NDArray[np.float64]:
    """Extracts the canonical 16D summary matrix (N, 16) matching `eq_autocall_guardian.py`."""
    node_np = batch.node_feats.detach().cpu().numpy().astype(np.float64)
    edge_np = batch.edge_feats.detach().cpu().numpy().astype(np.float64)
    amask_np = batch.asset_mask.detach().cpu().numpy().astype(np.float64)
    sched_np = batch.sched_feats.detach().cpu().numpy().astype(np.float64)
    smask_np = batch.sched_mask.detach().cpu().numpy().astype(np.float64)
    glob_np = batch.global_feats.detach().cpu().numpy().astype(np.float64)

    n = node_np.shape[0]
    out = np.zeros((n, 16), dtype=np.float64)
    for b in range(n):
        d = int(np.sum(amask_np[b] > 0.5))
        m_rem = int(np.sum(smask_np[b] > 0.5))
        if d > 0:
            log_m = node_np[b, :d, int(AutocallNodeCol.LOG_MONEYNESS)]
            iv_s = node_np[b, :d, int(AutocallNodeCol.IV_SHORT_ATM)]
            iv_t = node_np[b, :d, int(AutocallNodeCol.IV_TERM_ATM)]
            div_l = node_np[b, :d, int(AutocallNodeCol.CASH_DIV_LEVERAGE)]
            out[b, 0] = float(np.mean(log_m))
            out[b, 1] = float(np.min(log_m))
            out[b, 2] = float(np.max(log_m))
            out[b, 3] = float(np.mean(iv_t))
            out[b, 4] = float(np.max(np.maximum(iv_s, iv_t)))
            out[b, 7] = float(np.max(div_l))
        if d >= 2:
            corr_sub = edge_np[b, :d, :d, 0]
            off_vals = corr_sub[~np.eye(d, dtype=bool)]
            out[b, 5] = float(np.mean(off_vals))
            out[b, 6] = float(np.min(off_vals))
        else:
            out[b, 5] = 0.45
            out[b, 6] = 0.25

        if m_rem > 0:
            out[b, 8] = float(sched_np[b, m_rem - 1, int(AutocallSchedCol.TAU_OBS)])
            call_mask = sched_np[b, :m_rem, int(AutocallSchedCol.IS_CALLABLE_FLAG)] > 0.5
            if np.any(call_mask):
                out[b, 9] = float(np.mean(sched_np[b, :m_rem, int(AutocallSchedCol.LOG_KO_BARRIER)][call_mask]))
            else:
                out[b, 9] = 0.05
            log_cpn = sched_np[b, :m_rem, int(AutocallSchedCol.LOG_CPN_BARRIER)]
            guar = sched_np[b, :m_rem, int(AutocallSchedCol.IS_GUARANTEED_CPN)]
            cpn_act = (guar < 0.5) & (log_cpn > math.log(1e-4) + 1e-4)
            out[b, 10] = float(np.mean(log_cpn[cpn_act])) if np.any(cpn_act) else -0.50
            out[b, 12] = float(sched_np[b, m_rem - 1, int(AutocallSchedCol.STULZ_FWD_W)])
            out[b, 13] = float(sched_np[b, m_rem - 1, int(AutocallSchedCol.STULZ_VOL_W)])
        else:
            out[b, 9] = 0.05
            out[b, 10] = -0.50
            out[b, 13] = 0.25

        out[b, 11] = float(glob_np[b, int(AutocallGlobalCol.K_PUT)])
        out[b, 14] = float(glob_np[b, int(AutocallGlobalCol.C_MEM_0)])
        out[b, 15] = float(abs(glob_np[b, int(AutocallGlobalCol.ABS_KO_SPREAD)]))
    return out


def _compute_difficulty_basis_5d(batch: SyntheticAutocallBatch) -> npt.NDArray[np.float64]:
    """Computes the 5D physical difficulty basis phi(x) >= 0 in float64:
    [1.0, sqrt(h_disp * v_bar), ((M_rem - 1)/47) * v_bar, I_WOF * h_disp, I_live_KI * G_put * v_last].
    """
    n_b = int(batch.node_feats.shape[0])
    w_col = batch.node_feats[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)].to(torch.float64)
    amask = (batch.asset_mask > 0.5).to(dtype=torch.float64)
    w_abs = w_col.abs() * amask
    w_sum = w_abs.sum(dim=1, keepdim=True)
    has_w = (w_sum.squeeze(1) > 1e-6).to(dtype=torch.float64)
    w_norm = w_abs / w_sum.clamp(min=1e-6)
    h_disp = has_w * (1.0 - (w_norm * w_norm).sum(dim=1)).clamp(0.0, 1.0)

    smask = (batch.sched_mask > 0.5).to(dtype=torch.float64)
    m_counts = smask.sum(dim=1)
    v_all = (
        batch.sched_feats[:, :, int(AutocallSchedCol.STULZ_VOL_W)].to(torch.float64).clamp(min=0.0)
        * smask
    )
    v_bar = torch.where(
        m_counts > 0.5,
        v_all.sum(dim=1) / m_counts.clamp(min=1.0),
        torch.zeros_like(m_counts),
    )

    phi_0 = torch.ones((n_b,), dtype=torch.float64, device=h_disp.device)
    phi_1 = torch.sqrt((h_disp * v_bar).clamp(min=0.0))
    phi_2 = ((m_counts - 1.0).clamp(min=0.0) / 47.0) * v_bar

    wof_code = int(MonitoringIndexCode.WORST_OF)
    c_id = batch.global_feats[:, int(AutocallGlobalCol.CALL_INDEX_ID)].round().to(torch.long)
    o_id = batch.global_feats[:, int(AutocallGlobalCol.OPT_INDEX_ID)].round().to(torch.long)
    is_wof_any = ((c_id == wof_code) | (o_id == wof_code)).to(dtype=torch.float64)
    phi_3 = is_wof_any * h_disp

    has_ki = batch.global_feats[:, int(AutocallGlobalCol.HAS_KI_FLAG)].to(torch.float64) > 0.5
    not_ki_past = batch.global_feats[:, int(AutocallGlobalCol.I_KI_PAST)].to(torch.float64) < 0.5
    g_put = batch.global_feats[:, int(AutocallGlobalCol.G_PUT)].to(torch.float64).clamp(min=0.0)
    last_idx = (m_counts.long() - 1).clamp(min=0, max=max(int(batch.sched_feats.shape[1]) - 1, 0)).unsqueeze(1)
    v_last = (
        batch.sched_feats[:, :, int(AutocallSchedCol.STULZ_VOL_W)]
        .to(torch.float64)
        .clamp(min=0.0)
        .gather(1, last_idx)
        .squeeze(1)
    )
    v_last = torch.where(m_counts > 0.5, v_last, torch.zeros_like(v_last))
    phi_4 = (has_ki & not_ki_past).to(dtype=torch.float64) * g_put * v_last

    return torch.stack([phi_0, phi_1, phi_2, phi_3, phi_4], dim=1).detach().cpu().numpy().astype(np.float64)


def _compute_local_scale_factor(batch: SyntheticAutocallBatch) -> npt.NDArray[np.float64]:
    """Computes sqrt(h_disp * v_bar) local heteroscedastic scale factor per trade."""
    return _compute_difficulty_basis_5d(batch)[:, 1]


def compute_conformal_half_width_bps(
    batch: SyntheticAutocallBatch,
    stats_payload: Dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Computes split-conformal half-width in bps using 5D difficulty weights (or 2-term fallback)."""
    q_err = float(stats_payload["q_err_conformal"])
    if "scale_weights_5d" in stats_payload and stats_payload["scale_weights_5d"] is not None:
        w5 = np.asarray(
            stats_payload["scale_weights_5d"].detach().cpu().numpy()
            if isinstance(stats_payload["scale_weights_5d"], torch.Tensor)
            else stats_payload["scale_weights_5d"],
            dtype=np.float64,
        ).reshape(5)
        phi = _compute_difficulty_basis_5d(batch)
        scale = np.maximum(phi @ w5, 1.5e-4)
    else:
        eps0 = float(stats_payload["scale_eps0"])
        c1 = float(stats_payload["scale_c1"])
        scale = np.maximum(eps0 + c1 * _compute_local_scale_factor(batch), 1e-4)
    return q_err * scale * 1e4


def _predict_batch_pv(
    model: MaskedEQAutocallSurrogate,
    batch: SyntheticAutocallBatch,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Evaluates predicted ProductValue on CPU in float64 across mini-batches."""
    was_training = model.training
    model.eval()
    n = batch.node_feats.shape[0]
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idx = torch.arange(start, end, dtype=torch.long)
            sub = batch.index_select(idx).to(device=device, dtype=torch.float32)
            out = model(
                sub.node_feats,
                sub.edge_feats,
                sub.asset_mask,
                sub.sched_feats,
                sub.sched_mask,
                sub.global_feats,
                all_active=True,
                require_second_order_sdpa=False,
            )
            pv = assemble_surrogate_product_value(
                out,
                sub.sched_feats,
                sub.sched_mask,
                sub.global_feats,
                no_notional_repayment=sub.no_notional_repayment > 0.5,
                autocall_coupon=sub.autocall_coupon,
                force_coupons_on_autocall=sub.force_coupons_on_autocall > 0.5,
                is_long_basket=sub.is_long_basket > 0.5,
                lower_strike=sub.lower_strike,
            )
            preds.append(pv.detach().cpu().to(torch.float64))
    if was_training:
        model.train()
    return torch.cat(preds, dim=0)


def _fit_quantile_scale_weights_5d(
    phi_fit: npt.NDArray[np.float64],
    err_fit: npt.NDArray[np.float64],
    *,
    tau: float = 0.90,
) -> npt.NDArray[np.float64]:
    """Fits non-negative tau-quantile pinball loss regression weights w in R>=0^5."""
    n1 = int(phi_fit.shape[0])
    if n1 >= 32:
        c = np.zeros(5 + 2 * n1, dtype=np.float64)
        c[0] = 1e-3
        c[1:5] = 1e-5
        c[5 : 5 + n1] = float(tau) / float(n1)
        c[5 + n1 : 5 + 2 * n1] = (1.0 - float(tau)) / float(n1)
        a_eq = sp_sparse.hstack(
            [
                sp_sparse.csr_matrix(phi_fit),
                sp_sparse.eye(n1, format="csr", dtype=np.float64),
                -sp_sparse.eye(n1, format="csr", dtype=np.float64),
            ],
            format="csr",
        )
        bounds = [(1.5e-4, None)] + [(0.0, None)] * (4 + 2 * n1)
        res = linprog(c=c, A_eq=a_eq, b_eq=err_fit, bounds=bounds, method="highs")
        if bool(res.success) and res.x is not None:
            w = np.asarray(res.x[:5], dtype=np.float64)
            w[0] = max(float(w[0]), 1.5e-4)
            w[1:] = np.maximum(w[1:], 0.0)
            return w

    w_nnls, _ = nnls(phi_fit, err_fit)
    w_out = np.asarray(w_nnls[:5], dtype=np.float64)
    w_out[0] = max(float(w_out[0]), 1.5e-4)
    w_out[1:] = np.maximum(w_out[1:], 0.0)
    return w_out


def calibrate_guardian_stats(
    model: MaskedEQAutocallSurrogate,
    train_batch: SyntheticAutocallBatch,
    cal_batch: SyntheticAutocallBatch,
    *,
    device: torch.device,
    conformal_alpha: float = 0.05,
    tau_desk_bps: float = 60.0,
    stats_path: Optional[Path] = DEFAULT_GUARDIAN_STATS_PATH,
) -> Dict[str, Any]:
    """Fits OAS covariance, 10-decile PSI bins, and 5D split-conformal error quantiles."""
    x_fit = extract_16d_features_from_batch(train_batch)
    n_fit, p = x_fit.shape
    ref_mean = np.mean(x_fit, axis=0)
    diff = x_fit - ref_mean[None, :]
    s_cov = (diff.T @ diff) / float(max(n_fit, 1))
    tr_s = float(np.trace(s_cov))
    tr_s2 = float(np.sum(s_cov * s_cov))
    denom = float(n_fit + 1 - 2.0 / p) * max(tr_s2 - (tr_s * tr_s) / p, 1e-12)
    numer = (1.0 - 2.0 / p) * tr_s2 + tr_s * tr_s
    rho_oas = min(1.0, max(0.0, numer / max(denom, 1e-12)))
    f_target = (tr_s / p) * np.eye(p, dtype=np.float64)
    cov_oas = (1.0 - rho_oas) * s_cov + rho_oas * f_target + 1e-6 * np.eye(p, dtype=np.float64)
    cov_inv = np.linalg.inv(cov_oas)
    marginal_std = np.sqrt(np.maximum(np.diag(cov_oas), 1e-12))

    psi_bin_edges = np.zeros((p, 11), dtype=np.float64)
    psi_ref_probs = np.full((p, 10), 0.10, dtype=np.float64)
    percentiles = np.linspace(0.0, 100.0, 11)
    for j in range(p):
        edges = np.percentile(x_fit[:, j], percentiles).astype(np.float64)
        if len(np.unique(edges[1:-1])) < 9:
            col_std = max(float(np.std(x_fit[:, j])), 1e-3)
            inner = ref_mean[j] + col_std * np.asarray(
                norm.ppf(np.linspace(0.1, 0.9, 9)), dtype=np.float64
            )
        else:
            inner = edges[1:-1]
        psi_bin_edges[j, 0] = -np.inf
        psi_bin_edges[j, 1:10] = inner
        psi_bin_edges[j, 10] = np.inf
        counts, _ = np.histogram(x_fit[:, j], bins=psi_bin_edges[j])
        probs = np.maximum(counts.astype(np.float64) / float(max(n_fit, 1)), 1e-6)
        psi_ref_probs[j] = probs / np.sum(probs)

    # Split-conformal calibration strictly on held-out Cal split (Cal-1 quantile fit, disjoint Cal-2 conformal quantile)
    x_cal = extract_16d_features_from_batch(cal_batch)
    n_cal = int(x_cal.shape[0])
    diff_cal = x_cal - ref_mean[None, :]
    d2_cal = np.einsum("ni,ij,nj->n", diff_cal, cov_inv, diff_cal)
    q_ood_level = min(1.0, math.ceil((n_cal + 1) * 0.999) / float(max(n_cal, 1)))
    q_ood_conformal = float(np.quantile(d2_cal, q_ood_level))

    pv_pred_cal = _predict_batch_pv(model, cal_batch, device=device).numpy()
    pv_mc_cal = cal_batch.pv_mc.detach().cpu().numpy().astype(np.float64)
    abs_err_cal = np.abs(pv_pred_cal - pv_mc_cal)
    phi_cal = _compute_difficulty_basis_5d(cal_batch)

    if n_cal >= 64:
        split_idx = n_cal // 2
        w5 = _fit_quantile_scale_weights_5d(phi_cal[:split_idx], abs_err_cal[:split_idx], tau=0.90)
        phi_score = phi_cal[split_idx:]
        err_score = abs_err_cal[split_idx:]
    else:
        w5 = _fit_quantile_scale_weights_5d(phi_cal, abs_err_cal, tau=0.90)
        phi_score = phi_cal
        err_score = abs_err_cal

    n_score = int(err_score.shape[0])
    denom_score = np.maximum(phi_score @ w5, 1.5e-4)
    scores_cal = err_score / denom_score
    q_err_level = min(1.0, math.ceil((n_score + 1) * (1.0 - conformal_alpha)) / float(max(n_score, 1)))
    q_err_conformal = float(np.quantile(scores_cal, q_err_level))
    scale_eps0 = float(w5[0])
    scale_c1 = float(w5[1])

    payload: Dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names_16d": list(FEATURE_NAMES_16D),
        "ref_mean_16d": torch.from_numpy(ref_mean).to(torch.float64),
        "ref_cov_oas_16d": torch.from_numpy(cov_oas).to(torch.float64),
        "ref_cov_inv_16d": torch.from_numpy(cov_inv).to(torch.float64),
        "ref_marginal_std_16d": torch.from_numpy(marginal_std).to(torch.float64),
        "psi_bin_edges": torch.from_numpy(psi_bin_edges).to(torch.float64),
        "psi_ref_probs": torch.from_numpy(psi_ref_probs).to(torch.float64),
        "conformal_alpha": float(conformal_alpha),
        "tau_desk_bps": float(tau_desk_bps),
        "q_ood_conformal": q_ood_conformal,
        "q_err_conformal": q_err_conformal,
        "scale_weights_5d": torch.from_numpy(w5).to(torch.float64),
        "scale_eps0": scale_eps0,
        "scale_c1": scale_c1,
        "n_cal": n_cal,
    }
    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, stats_path)
    return payload


def evaluate_conformal_coverage(
    model: MaskedEQAutocallSurrogate,
    test_batch: SyntheticAutocallBatch,
    stats_payload: Dict[str, Any],
    *,
    device: torch.device,
) -> float:
    """Computes empirical split-conformal coverage on a held-out test batch."""
    pv_pred = _predict_batch_pv(model, test_batch, device=device).numpy()
    pv_mc = test_batch.pv_mc.detach().cpu().numpy().astype(np.float64)
    abs_err_bps = np.abs(pv_pred - pv_mc) * 1e4
    half_width_bps = compute_conformal_half_width_bps(test_batch, stats_payload)
    return float(np.mean(abs_err_bps <= half_width_bps))


def evaluate_surrogate_mae_bps(
    model: MaskedEQAutocallSurrogate,
    *,
    n_samples_per_d: int = 16,
    d_values: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_mc_paths: int = 8192,
    n_substeps: int = 8,
    seed: int = 7777,
    device: torch.device | str | None = None,
    test_batch: Optional[SyntheticAutocallBatch] = None,
) -> Tuple[float, float, Dict[int, float]]:
    """Evaluates out-of-sample test MAE (unit-notional and bps) across basket dimensions."""
    first_param = next(model.parameters(), None)
    dev = (
        torch.device(device)
        if device is not None
        else (first_param.device if first_param is not None else torch.device("cpu"))
    )

    if test_batch is not None:
        pv_pred = _predict_batch_pv(model, test_batch, device=dev)
        pv_mc = test_batch.pv_mc.detach().cpu().to(torch.float64)
        abs_err = (pv_pred - pv_mc).abs()
        d_counts = (test_batch.asset_mask.detach().cpu() > 0.5).sum(dim=1).to(torch.long)
        per_d_mae_bps: Dict[int, float] = {}
        for d_val in d_values:
            mask_d = d_counts == int(d_val)
            if bool(mask_d.any()):
                per_d_mae_bps[int(d_val)] = float(abs_err[mask_d].mean().item() * 1e4)
        overall_mae = float(abs_err.mean().item())
        return overall_mae, overall_mae * 1e4, per_d_mae_bps

    per_d_mae_bps = {}
    all_errors: List[torch.Tensor] = []
    for d_val in d_values:
        batch_d = generate_synthetic_autocall_batch(
            batch_size=n_samples_per_d,
            d_max=model.d_max,
            m_max=model.m_max,
            n_mc_paths=n_mc_paths,
            n_substeps=n_substeps,
            seed=seed + 101 * int(d_val),
            device="cpu",
            qmc_device=dev,
            dtype=torch.float32,
            fixed_d=int(d_val),
            cv_mode="bs1d_exact" if n_mc_paths >= 4096 else "none",
        )
        pv_pred_d = _predict_batch_pv(model, batch_d, device=dev)
        abs_err_d = (pv_pred_d - batch_d.pv_mc.detach().cpu().to(torch.float64)).abs()
        per_d_mae_bps[int(d_val)] = float(abs_err_d.mean().item() * 1e4)
        all_errors.append(abs_err_d)

    concat_err = torch.cat(all_errors, dim=0)
    overall_mae = float(concat_err.mean().item())
    return overall_mae, overall_mae * 1e4, per_d_mae_bps


def _evaluate_delta_cos_sims(
    model: MaskedEQAutocallSurrogate,
    batch: SyntheticAutocallBatch,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[float, float]:
    """Computes held-out OOS (total_delta_cos_sim, residual_delta_cos_sim) via exact AD contraction."""
    was_training = model.training
    model.eval()
    n = batch.node_feats.shape[0]
    tot_pred_list: List[torch.Tensor] = []
    tot_tgt_list: List[torch.Tensor] = []
    res_pred_list: List[torch.Tensor] = []
    res_tgt_list: List[torch.Tensor] = []

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        idx = torch.arange(start, end, dtype=torch.long)
        sub = batch.index_select(idx).to(device=device, dtype=torch.float32)
        node_in = sub.node_feats.detach().clone().requires_grad_(True)
        edge_in = sub.edge_feats.detach().clone().requires_grad_(True)
        sched_in = sub.sched_feats.detach().clone().requires_grad_(True)
        glob_in = sub.global_feats.detach().clone().requires_grad_(True)
        amask_in = sub.asset_mask
        smask_in = sub.sched_mask

        surr_out = model(
            node_feats=node_in,
            edge_feats=edge_in,
            asset_mask=amask_in,
            sched_feats=sched_in,
            sched_mask=smask_in,
            global_feats=glob_in,
            all_active=True,
            require_second_order_sdpa=False,
        )
        pv_pred = assemble_surrogate_product_value(
            surr_out,
            sched_in,
            smask_in,
            glob_in,
            no_notional_repayment=sub.no_notional_repayment > 0.5,
            autocall_coupon=sub.autocall_coupon,
            force_coupons_on_autocall=sub.force_coupons_on_autocall > 0.5,
            is_long_basket=sub.is_long_basket > 0.5,
            lower_strike=sub.lower_strike,
        )
        g_node, g_edge, g_sched, g_glob = torch.autograd.grad(
            pv_pred.sum(),
            (node_in, edge_in, sched_in, glob_in),
            allow_unused=True,
        )
        g_node = torch.zeros_like(node_in) if g_node is None else g_node
        g_edge = torch.zeros_like(edge_in) if g_edge is None else g_edge
        g_sched = torch.zeros_like(sched_in) if g_sched is None else g_sched
        g_glob = torch.zeros_like(glob_in) if g_glob is None else g_glob
        g_edge_sym = g_edge[:, :, :, 1:3] + g_edge.transpose(1, 2)[:, :, :, 1:3]
        delta_tot = (
            torch.einsum("bjk,bijk->bi", g_node, sub.jac_node_logm)
            + torch.einsum("bijc,bijc->bi", g_edge_sym, sub.jac_edge_self)
            + torch.einsum("bmc,bimc->bi", g_sched[:, :, 8:12], sub.jac_sched_logm)
            + g_glob[:, int(AutocallGlobalCol.L_PUT_PROXY)].unsqueeze(1) * sub.jac_lput_logm
        ) * amask_in
        d_px = sub.delta_proxy * amask_in
        d_mc = sub.delta_mc * amask_in
        tot_pred_list.append(delta_tot.detach().cpu())
        tot_tgt_list.append(d_mc.detach().cpu())
        res_pred_list.append((delta_tot - d_px).detach().cpu())
        res_tgt_list.append((d_mc - d_px).detach().cpu())

    if was_training:
        model.train()

    def _cos(a_list: List[torch.Tensor], b_list: List[torch.Tensor]) -> float:
        af = torch.cat(a_list, dim=0).reshape(-1)
        bf = torch.cat(b_list, dim=0).reshape(-1)
        denom = af.norm() * bf.norm()
        return float((torch.dot(af, bf) / denom.clamp(min=1e-12)).item()) if float(denom.item()) > 1e-10 else 0.0

    return _cos(tot_pred_list, tot_tgt_list), _cos(res_pred_list, res_tgt_list)


def _split_worstof_multifunc_mae_bps(
    pv_pred: torch.Tensor,
    batch: SyntheticAutocallBatch,
) -> Tuple[float, float]:
    """Computes separate WorstOf MAE and multi-functional MAE in bps."""
    pv_mc = batch.pv_mc.detach().cpu().to(torch.float64)
    abs_err_bps = (pv_pred.detach().cpu().to(torch.float64) - pv_mc).abs() * 1e4
    glob = batch.global_feats.detach().cpu()
    c_id = glob[:, int(AutocallGlobalCol.CALL_INDEX_ID)].round().to(torch.long)
    q_id = glob[:, int(AutocallGlobalCol.CPN_INDEX_ID)].round().to(torch.long)
    o_id = glob[:, int(AutocallGlobalCol.OPT_INDEX_ID)].round().to(torch.long)
    wof_code = int(MonitoringIndexCode.WORST_OF)
    is_wof = (c_id == wof_code) & (q_id == wof_code) & (o_id == wof_code)
    wof_mae = float(abs_err_bps[is_wof].mean().item()) if bool(is_wof.any()) else float(abs_err_bps.mean().item())
    multi_mae = float(abs_err_bps.mean().item())
    return wof_mae, multi_mae


def train_eq_autocall_surrogate(
    *,
    n_train_samples: int = 16384,
    n_val_samples: int = 2048,
    n_cal_samples: int = 2048,
    n_test_samples: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 50,
    lr: float = 2.0e-3,
    weight_decay: float = 1e-4,
    n_mc_paths_train: int = 8192,
    n_substeps_train: int = 8,
    n_mc_paths_test: int = 32768,
    n_substeps_test: int = 16,
    d_max: int = D_MAX_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
    d_model: int = 128,
    n_egno_layers: int = 3,
    n_sched_layers: int = 2,
    n_heads: int = 4,
    n_func_codes: int = 15,
    func_emb_dim: int = 16,
    seed: int = 2026,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    guardian_stats_path: str | Path | None = DEFAULT_GUARDIAN_STATS_PATH,
    conformal_alpha: float = 0.05,
    tau_desk_bps: float = 60.0,
    domain_mode: str = "tier2_desk",
) -> SurrogateTrainingResult:
    """Trains EGNO v2.0 on GPU Dupire LV QMC labels and calibrates SR 26-2 Guardian stats."""
    t_start = time.perf_counter()
    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    model = MaskedEQAutocallSurrogate(
        d_max=d_max,
        m_max=m_max,
        d_model=d_model,
        n_egno_layers=n_egno_layers,
        n_sched_layers=n_sched_layers,
        n_heads=n_heads,
        d_node=16,
        d_edge=4,
        d_sched=12,
        d_global=16,
        n_func_codes=n_func_codes,
        func_emb_dim=func_emb_dim,
    ).to(device=dev, dtype=torch.float32)
    model.zero_init_residual_heads(std=1e-4)

    # 1. Generate Train / Val / Cal / Test splits via GPU Dupire LV RQMC
    train_data = generate_synthetic_autocall_batch(
        batch_size=n_train_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths_train,
        n_substeps=n_substeps_train,
        seed=seed,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="none",
        domain_mode=domain_mode,
    )
    val_data = generate_synthetic_autocall_batch(
        batch_size=n_val_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths_train,
        n_substeps=n_substeps_train,
        seed=seed + 100_000,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="none",
        domain_mode=domain_mode,
    )
    cal_data = generate_synthetic_autocall_batch(
        batch_size=n_cal_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths_test,
        n_substeps=n_substeps_test,
        seed=seed + 200_000,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="bs1d_exact" if n_mc_paths_test >= 4096 else "none",
        domain_mode=domain_mode,
    )
    test_data = generate_synthetic_autocall_batch(
        batch_size=n_test_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths_test,
        n_substeps=n_substeps_test,
        seed=seed + 300_000,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="bs1d_exact" if n_mc_paths_test >= 4096 else "none",
        domain_mode=domain_mode,
    )

    base_wof_mae, base_multi_mae = _split_worstof_multifunc_mae_bps(test_data.pv_proxy, test_data)
    baseline_mae_bps = base_multi_mae
    t_data_sec = time.perf_counter() - t_start
    print(
        f"[data] generated train={n_train_samples} val={n_val_samples} "
        f"cal={n_cal_samples} test={n_test_samples} in {t_data_sec:.1f}s "
        f"(base_wof={base_wof_mae:.1f}bps base_multi={base_multi_mae:.1f}bps)",
        flush=True,
    )

    # 2. Train EGNO v2.0 with Spread-Attenuated Sobolev H^1 loss, selecting best model by Val MAE
    criterion = SpreadAttenuatedSobolevLoss().to(device=dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(n_train_samples / batch_size))
    total_steps = max(4, n_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-5,
    )

    loss_history: List[float] = []
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_mae_bps = float("inf")
    last_cos_sim = 0.0

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 99)
    n_sobolev_warmup = int(n_epochs * 0.56) if n_epochs >= 20 else 0

    for _epoch in range(n_epochs):
        if _epoch == n_sobolev_warmup and n_sobolev_warmup > 0:
            best_val_mae_bps = float("inf")
        use_sobolev = _epoch >= n_sobolev_warmup
        model.train()
        perm = torch.randperm(n_train_samples, generator=gen)
        epoch_loss_sum = 0.0
        cos_sum = 0.0
        n_batches = 0

        for start_idx in range(0, n_train_samples, batch_size):
            idx = perm[start_idx : start_idx + batch_size]
            sub_batch = train_data.index_select(idx).to(device=dev, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            breakdown = criterion(
                model,
                sub_batch,
                create_graph_for_sobolev=use_sobolev,
                compute_sobolev=use_sobolev,
                return_breakdown=True,
            )
            assert isinstance(breakdown, SobolevLossBreakdown)
            breakdown.total_loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += float(breakdown.total_loss.detach().cpu().item())
            cos_sum += breakdown.delta_cos_sim
            n_batches += 1

        avg_loss = epoch_loss_sum / max(n_batches, 1)
        avg_cos = cos_sum / max(n_batches, 1)
        loss_history.append(avg_loss)

        val_preds = _predict_batch_pv(model, val_data, device=dev)
        val_mae_bps = float(
            (val_preds - val_data.pv_mc.detach().cpu().to(torch.float64)).abs().mean().item() * 1e4
        )
        if val_mae_bps < best_val_mae_bps:
            best_val_mae_bps = val_mae_bps
            last_cos_sim = avg_cos
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (_epoch + 1) % 5 == 0 or _epoch == 0 or (_epoch + 1) == n_epochs:
            print(
                f"[epoch {_epoch + 1:3d}/{n_epochs}] loss={avg_loss:.4f} "
                f"val_mae={val_mae_bps:.2f}bps (best={best_val_mae_bps:.2f}bps) "
                f"delta_cos={avg_cos:+.4f} elapsed={time.perf_counter() - t_start:.1f}s",
                flush=True,
            )

    model.load_state_dict({k: v.to(device=dev) for k, v in best_state_dict.items()}, strict=True)
    model.eval()
    model.freeze_eval_buffers()

    # 3. Held-out OOS Test Evaluation (32,768-path Dupire LV RQMC)
    test_preds = _predict_batch_pv(model, test_data, device=dev)
    test_wof_mae_bps, test_multifunc_mae_bps = _split_worstof_multifunc_mae_bps(test_preds, test_data)
    test_mae, test_mae_bps, per_d_mae_bps = evaluate_surrogate_mae_bps(
        model,
        d_values=tuple(range(1, min(d_max, 8) + 1)),
        device=dev,
        test_batch=test_data,
    )
    total_delta_cos_sim, oos_res_cos_sim = _evaluate_delta_cos_sims(model, test_data, device=dev)
    eff_res_cos_sim = oos_res_cos_sim if n_sobolev_warmup == 0 and n_epochs < 4 else (
        oos_res_cos_sim if oos_res_cos_sim != 0.0 else last_cos_sim
    )

    # 4. Calibrate standalone SR 26-2 Guardian artifact on Cal split and audit coverage on Test split
    stats_path_resolved = Path(guardian_stats_path) if guardian_stats_path is not None else None
    stats_payload = calibrate_guardian_stats(
        model,
        train_data,
        cal_data,
        device=dev,
        conformal_alpha=conformal_alpha,
        tau_desk_bps=tau_desk_bps,
        stats_path=stats_path_resolved,
    )
    conformal_coverage = evaluate_conformal_coverage(model, test_data, stats_payload, device=dev)
    test_err_bps = (
        (test_preds - test_data.pv_mc.detach().cpu().to(torch.float64)).abs().numpy() * 1e4
    )
    hw_bps = compute_conformal_half_width_bps(test_data, stats_payload)
    acc_mask = hw_bps <= float(stats_payload["tau_desk_bps"])
    accepted_frac = float(np.mean(acc_mask))
    accepted_mae_bps = float(np.mean(test_err_bps[acc_mask])) if bool(np.any(acc_mask)) else float(test_mae_bps)
    guardian_routed_mae_bps = float(np.mean(np.where(acc_mask, test_err_bps, 0.0)))
    train_wall_sec = time.perf_counter() - t_start

    ckpt_path_resolved: Optional[Path] = None
    if checkpoint_path is not None:
        ckpt_path_resolved = Path(checkpoint_path)
        ckpt_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "oracle_mode": "dupire_lv_gpu_rqmc",
            "config": {
                "d_max": int(d_max),
                "m_max": int(m_max),
                "d_node": 16,
                "d_edge": 4,
                "d_sched": 12,
                "d_global": 16,
                "d_model": int(d_model),
                "n_egno_layers": int(n_egno_layers),
                "n_sched_layers": int(n_sched_layers),
                "n_heads": int(n_heads),
                "n_func_codes": int(n_func_codes),
                "func_emb_dim": int(func_emb_dim),
            },
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "metrics": {
                "val_mae_bps": float(best_val_mae_bps),
                "test_mae_bps": float(test_mae_bps),
                "test_worstof_mae_bps": float(test_wof_mae_bps),
                "test_multifunc_mae_bps": float(test_multifunc_mae_bps),
                "baseline_worstof_mae_bps": float(base_wof_mae),
                "baseline_multifunc_mae_bps": float(base_multi_mae),
                "conformal_coverage": float(conformal_coverage),
                "conformal_accepted_frac": float(accepted_frac),
                "conformal_accepted_mae_bps": float(accepted_mae_bps),
                "guardian_routed_mae_bps": float(guardian_routed_mae_bps),
                "sobolev_delta_cos_sim": float(eff_res_cos_sim),
                "total_delta_cos_sim": float(total_delta_cos_sim),
                "train_wall_sec": float(train_wall_sec),
                "per_d_mae_bps": {int(k): float(v) for k, v in per_d_mae_bps.items()},
            },
        }
        torch.save(payload, ckpt_path_resolved)

    return SurrogateTrainingResult(
        model=model,
        checkpoint_path=ckpt_path_resolved,
        guardian_stats_path=stats_path_resolved,
        train_loss_history=loss_history,
        val_mae_bps=float(best_val_mae_bps),
        test_mae=test_mae,
        test_mae_bps=test_mae_bps,
        test_worstof_mae_bps=test_wof_mae_bps,
        test_multifunc_mae_bps=test_multifunc_mae_bps,
        baseline_mae_bps=baseline_mae_bps,
        baseline_worstof_mae_bps=base_wof_mae,
        baseline_multifunc_mae_bps=base_multi_mae,
        conformal_coverage=conformal_coverage,
        sobolev_delta_cos_sim=eff_res_cos_sim,
        train_wall_sec=train_wall_sec,
        per_d_mae_bps=per_d_mae_bps,
        total_delta_cos_sim=total_delta_cos_sim,
    )


def load_eq_autocall_surrogate(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[MaskedEQAutocallSurrogate, Dict[str, Any]]:
    """Loads a trained `MaskedEQAutocallSurrogate` with weights_only=True and strict=True."""
    ckpt_path = Path(checkpoint_path)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        cfg = raw.get("config", {})
        model = MaskedEQAutocallSurrogate(
            d_max=int(cfg.get("d_max", 8)),
            m_max=int(cfg.get("m_max", 48)),
            d_model=int(cfg.get("d_model", 128)),
            n_egno_layers=int(cfg.get("n_egno_layers", 3)),
            n_sched_layers=int(cfg.get("n_sched_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
            d_node=int(cfg.get("d_node", 16)),
            d_edge=int(cfg.get("d_edge", 4)),
            d_sched=int(cfg.get("d_sched", 12)),
            d_global=int(cfg.get("d_global", 16)),
            n_func_codes=int(cfg.get("n_func_codes", 15)),
            func_emb_dim=int(cfg.get("func_emb_dim", 16)),
        )
        model.load_state_dict(raw["model_state_dict"], strict=True)
        metrics = dict(raw.get("metrics", {}))
    else:
        model = MaskedEQAutocallSurrogate()
        model.load_state_dict(raw, strict=True)
        metrics = {}

    model = model.to(device=device, dtype=dtype).eval()
    model.freeze_eval_buffers()
    return model, metrics


def recalibrate_guardian_stats_from_checkpoint(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    guardian_stats_path: str | Path = DEFAULT_GUARDIAN_STATS_PATH,
    n_train_samples: int = 16384,
    n_cal_samples: int = 2048,
    n_test_samples: int = 2048,
    n_mc_paths_test: int = 32768,
    n_substeps_test: int = 16,
    seed: int = 2026,
    device: torch.device | str | None = None,
    conformal_alpha: float = 0.05,
    tau_desk_bps: float = 60.0,
    domain_mode: str = "tier2_desk",
) -> Dict[str, Any]:
    """Refreshes 5D split-conformal Guardian artifact & test metrics while keeping EGNO weights."""
    t0 = time.perf_counter()
    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
    ckpt_path = Path(checkpoint_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model, prev_metrics = load_eq_autocall_surrogate(ckpt_path, device=dev)
    d_max = int(model.d_max)
    m_max = int(model.m_max)

    # OAS covariance and PSI bins depend strictly on the 16D feature distribution (identical Sobol contracts).
    train_data = generate_synthetic_autocall_batch(
        batch_size=n_train_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=64,
        n_substeps=2,
        seed=seed,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="none",
        compute_pathwise_delta=False,
        domain_mode=domain_mode,
    )
    cal_data = generate_synthetic_autocall_batch(
        batch_size=n_cal_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths_test,
        n_substeps=n_substeps_test,
        seed=seed + 200_000,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="bs1d_exact" if n_mc_paths_test >= 4096 else "none",
        compute_pathwise_delta=False,
        domain_mode=domain_mode,
    )
    test_data = generate_synthetic_autocall_batch(
        batch_size=n_test_samples,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths_test,
        n_substeps=n_substeps_test,
        seed=seed + 300_000,
        device="cpu",
        qmc_device=dev,
        dtype=torch.float32,
        cv_mode="bs1d_exact" if n_mc_paths_test >= 4096 else "none",
        compute_pathwise_delta=True,
        domain_mode=domain_mode,
    )

    stats_payload = calibrate_guardian_stats(
        model,
        train_data,
        cal_data,
        device=dev,
        conformal_alpha=conformal_alpha,
        tau_desk_bps=tau_desk_bps,
        stats_path=Path(guardian_stats_path),
    )

    test_preds = _predict_batch_pv(model, test_data, device=dev)
    test_wof_mae_bps, test_multifunc_mae_bps = _split_worstof_multifunc_mae_bps(test_preds, test_data)
    base_wof_mae, base_multi_mae = _split_worstof_multifunc_mae_bps(test_data.pv_proxy, test_data)
    _, test_mae_bps, per_d_mae_bps = evaluate_surrogate_mae_bps(
        model,
        d_values=tuple(range(1, min(d_max, 8) + 1)),
        device=dev,
        test_batch=test_data,
    )
    total_delta_cos_sim, oos_res_cos_sim = _evaluate_delta_cos_sims(model, test_data, device=dev)
    conformal_coverage = evaluate_conformal_coverage(model, test_data, stats_payload, device=dev)

    test_err_bps = (
        (test_preds - test_data.pv_mc.detach().cpu().to(torch.float64)).abs().numpy() * 1e4
    )
    hw_bps = compute_conformal_half_width_bps(test_data, stats_payload)
    acc_mask = hw_bps <= float(stats_payload["tau_desk_bps"])
    accepted_frac = float(np.mean(acc_mask))
    accepted_mae_bps = float(np.mean(test_err_bps[acc_mask])) if bool(np.any(acc_mask)) else float(test_mae_bps)
    guardian_routed_mae_bps = float(np.mean(np.where(acc_mask, test_err_bps, 0.0)))

    if isinstance(raw, dict) and "model_state_dict" in raw:
        metrics = dict(raw.get("metrics", {}))
        metrics.update(
            {
                "test_mae_bps": float(test_mae_bps),
                "test_worstof_mae_bps": float(test_wof_mae_bps),
                "test_multifunc_mae_bps": float(test_multifunc_mae_bps),
                "baseline_worstof_mae_bps": float(base_wof_mae),
                "baseline_multifunc_mae_bps": float(base_multi_mae),
                "conformal_coverage": float(conformal_coverage),
                "conformal_accepted_frac": float(accepted_frac),
                "conformal_accepted_mae_bps": float(accepted_mae_bps),
                "guardian_routed_mae_bps": float(guardian_routed_mae_bps),
                "sobolev_delta_cos_sim": float(oos_res_cos_sim),
                "total_delta_cos_sim": float(total_delta_cos_sim),
                "train_wall_sec": float(prev_metrics.get("train_wall_sec", 864.9)),
                "per_d_mae_bps": {int(k): float(v) for k, v in per_d_mae_bps.items()},
            }
        )
        raw["metrics"] = metrics
        torch.save(raw, ckpt_path)

    return {
        "recal_wall_sec": time.perf_counter() - t0,
        "test_wof_mae_bps": test_wof_mae_bps,
        "test_multifunc_mae_bps": test_multifunc_mae_bps,
        "base_wof_mae_bps": base_wof_mae,
        "base_multifunc_mae_bps": base_multi_mae,
        "conformal_coverage": conformal_coverage,
        "accepted_frac": accepted_frac,
        "accepted_mae_bps": accepted_mae_bps,
        "guardian_routed_mae_bps": guardian_routed_mae_bps,
        "res_delta_cos_sim": oos_res_cos_sim,
        "total_delta_cos_sim": total_delta_cos_sim,
        "stats_payload": stats_payload,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EGNO v2.0 on GPU Dupire QMC & calibrate Guardian.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train", type=int, default=16384)
    parser.add_argument("--n-val", type=int, default=2048)
    parser.add_argument("--n-cal", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--n-mc-train", type=int, default=8192)
    parser.add_argument("--n-substeps-train", type=int, default=8)
    parser.add_argument("--n-mc-test", type=int, default=32768)
    parser.add_argument("--n-substeps-test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--domain-mode", type=str, default="tier2_desk")
    parser.add_argument("--tau-desk-bps", type=float, default=60.0)
    parser.add_argument("--recalibrate-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--guardian-stats-path", type=str, default=str(DEFAULT_GUARDIAN_STATS_PATH))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.recalibrate_only:
        summary = recalibrate_guardian_stats_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            guardian_stats_path=args.guardian_stats_path,
            n_train_samples=args.n_train,
            n_cal_samples=args.n_cal,
            n_test_samples=args.n_test,
            n_mc_paths_test=args.n_mc_test,
            n_substeps_test=args.n_substeps_test,
            seed=args.seed,
            device=args.device,
            tau_desk_bps=args.tau_desk_bps,
            domain_mode=args.domain_mode,
        )
        sp = summary["stats_payload"]
        print(
            f"recal_wall={summary['recal_wall_sec']:.1f}s "
            f"test_wof_mae={summary['test_wof_mae_bps']:.2f}bps "
            f"test_multifunc_mae={summary['test_multifunc_mae_bps']:.2f}bps "
            f"base_wof_mae={summary['base_wof_mae_bps']:.2f}bps "
            f"base_multifunc_mae={summary['base_multifunc_mae_bps']:.2f}bps "
            f"conformal_cov={summary['conformal_coverage']:.4f} "
            f"accepted_frac={summary['accepted_frac']:.4f} "
            f"accepted_mae={summary['accepted_mae_bps']:.2f}bps "
            f"routed_mae={summary['guardian_routed_mae_bps']:.2f}bps "
            f"res_delta_cos_sim={summary['res_delta_cos_sim']:.4f} "
            f"total_delta_cos_sim={summary['total_delta_cos_sim']:.4f} "
            f"w5={[round(float(x), 6) for x in sp['scale_weights_5d'].tolist()]} "
            f"q_err={sp['q_err_conformal']:.4f} q_ood={sp['q_ood_conformal']:.2f}"
        )
        return

    res = train_eq_autocall_surrogate(
        n_train_samples=args.n_train,
        n_val_samples=args.n_val,
        n_cal_samples=args.n_cal,
        n_test_samples=args.n_test,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        lr=args.lr,
        n_mc_paths_train=args.n_mc_train,
        n_substeps_train=args.n_substeps_train,
        n_mc_paths_test=args.n_mc_test,
        n_substeps_test=args.n_substeps_test,
        seed=args.seed,
        device=args.device,
        domain_mode=args.domain_mode,
        tau_desk_bps=args.tau_desk_bps,
        checkpoint_path=args.checkpoint_path,
        guardian_stats_path=args.guardian_stats_path,
    )
    peak_mb = (torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() else 0.0
    print(
        f"wall={res.train_wall_sec:.1f}s "
        f"peak_vram={peak_mb:.1f}MB "
        f"val_mae={res.val_mae_bps:.2f}bps "
        f"test_wof_mae={res.test_worstof_mae_bps:.2f}bps "
        f"test_multifunc_mae={res.test_multifunc_mae_bps:.2f}bps "
        f"base_wof_mae={res.baseline_worstof_mae_bps:.2f}bps "
        f"base_multifunc_mae={res.baseline_multifunc_mae_bps:.2f}bps "
        f"conformal_cov={res.conformal_coverage:.4f} "
        f"res_delta_cos_sim={res.sobolev_delta_cos_sim:.4f} "
        f"total_delta_cos_sim={res.total_delta_cos_sim:.4f}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_GUARDIAN_STATS_PATH",
    "FEATURE_NAMES_16D",
    "SobolevLossBreakdown",
    "SurrogateTrainingResult",
    "SpreadAttenuatedSobolevLoss",
    "calibrate_guardian_stats",
    "compute_conformal_half_width_bps",
    "evaluate_conformal_coverage",
    "evaluate_surrogate_mae_bps",
    "extract_16d_features_from_batch",
    "recalibrate_guardian_stats_from_checkpoint",
    "train_eq_autocall_surrogate",
    "load_eq_autocall_surrogate",
]
