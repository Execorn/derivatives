"""Sobolev H^1 trainer and SR 26-2 OAS/split-conformal calibrator for EGNO v2.0."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, fields
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse as sp_sparse
from scipy.optimize import linprog, minimize, nnls
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
from deepvol.mrm.eq_autocall_guardian import _regularize_cov_oas_correlation_space
from deepvol.surrogates.eq_autocall_egno import (
    MaskedEQAutocallSurrogate,
    assemble_surrogate_components,
    assemble_surrogate_product_value,
)

_LOGGER = logging.getLogger(__name__)


def _atomic_torch_save(payload: Any, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent),
        prefix=f".{dest.name}.tmp.",
        suffix=".pt",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _load_or_gen_cached_split(
    split_name: str,
    gen_fn: Callable[[], SyntheticAutocallBatch],
    *,
    data_dir: str | Path | None,
    n_samples: int,
    domain_mode: str,
    seed: int,
    d_max: int,
    m_max: int,
    save_on_miss: bool = True,
) -> SyntheticAutocallBatch:
    """Loads a cached `.pt` split when sample count (`>= n_samples`) and metadata match, else regenerates."""
    if data_dir is not None:
        split_path = Path(data_dir) / f"{split_name}.pt"
        if split_path.is_file():
            raw_split = torch.load(split_path, map_location="cpu", weights_only=True)
            if isinstance(raw_split, dict):
                batch_fields = fields(SyntheticAutocallBatch)
                required_fields = {
                    f.name for f in batch_fields if f.default is MISSING and f.default_factory is MISSING
                }
                valid_fields = {f.name for f in batch_fields}
                meta = raw_split.get("_meta")
                seed_ok = isinstance(meta, dict) and ("seed" not in meta or meta.get("seed") == int(seed))
                meta_ok = (
                    isinstance(meta, dict)
                    and meta.get("domain_mode") == str(domain_mode)
                    and seed_ok
                    and required_fields.issubset(raw_split.keys())
                )
                if meta_ok:
                    batch_kwargs = {k: v for k, v in raw_split.items() if k in valid_fields}
                    cached_batch = SyntheticAutocallBatch(**batch_kwargs)
                    n_cached = int(cached_batch.node_feats.shape[0])
                    if (
                        n_cached >= n_samples
                        and int(cached_batch.node_feats.shape[1]) == int(d_max)
                        and int(cached_batch.sched_feats.shape[1]) == int(m_max)
                    ):
                        if n_cached > n_samples:
                            return cached_batch.index_select(torch.arange(n_samples, dtype=torch.long))
                        return cached_batch
                _LOGGER.info(
                    "Regenerating cached split %s (expected n>=%d, domain_mode=%s, seed=%d, d_max=%d, m_max=%d; got meta=%r).",
                    split_path,
                    n_samples,
                    domain_mode,
                    seed,
                    d_max,
                    m_max,
                    meta,
                )
        batch_out = gen_fn()
        if save_on_miss:
            payload_split: Dict[str, Any] = {
                f.name: getattr(batch_out, f.name) for f in fields(SyntheticAutocallBatch)
            }
            payload_split["_meta"] = {
                "n_samples": int(batch_out.node_feats.shape[0]),
                "domain_mode": str(domain_mode),
                "seed": int(seed),
                "d_max": int(d_max),
                "m_max": int(m_max),
            }
            _atomic_torch_save(payload_split, split_path)
        return batch_out
    return gen_fn()

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


def _concat_batches(batches: List[SyntheticAutocallBatch]) -> SyntheticAutocallBatch:
    """Concatenates non-empty `SyntheticAutocallBatch` splits along dimension 0."""
    non_empty = [b for b in batches if int(b.node_feats.shape[0]) > 0]
    if not non_empty:
        return batches[0]
    if len(non_empty) == 1:
        return non_empty[0]
    merged: Dict[str, Any] = {}
    for f in fields(SyntheticAutocallBatch):
        vals = [getattr(b, f.name) for b in non_empty]
        if all(v is None for v in vals):
            merged[f.name] = None
        elif any(v is None for v in vals):
            merged[f.name] = None
        else:
            merged[f.name] = torch.cat(vals, dim=0)
    return SyntheticAutocallBatch(**merged)


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
        lambda_comp: float = 0.40,
        lambda_delta: float = 0.65,
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
        pv_coupon_pred: Optional[torch.Tensor] = None,
        pv_coupon_target: Optional[torch.Tensor] = None,
        coupon_pv_valid: Optional[torch.Tensor] = None,
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

        # 1. Balanced SmoothL1 + MSE value loss (targets <=15 bps MAE across all regimes)
        val_loss = (
            160.0 * F.smooth_l1_loss(pv_pred, pv_t, beta=0.0002)
            + 40.0 * F.mse_loss(pv_pred, pv_t)
        )

        # 2. Component probability, coupon-leg PV & put-loss supervision on live survival periods
        p_err_sq = ((p_pred - p_t) ** 2) * m_sched
        p_huber = F.smooth_l1_loss(p_pred, p_t, beta=0.005, reduction="none") * m_sched
        surv_prev_t = torch.cat(
            [torch.ones_like(p_t[:, :1]), (1.0 - torch.cumsum(p_t, dim=1))[:, :-1]],
            dim=1,
        )
        q_valid = m_sched * (surv_prev_t > 1e-4).to(dtype=dtype)
        q_count = q_valid.sum(dim=1).clamp(min=1.0)
        q_err_sq = ((q_pred - q_t) ** 2) * q_valid
        q_huber = F.smooth_l1_loss(q_pred, q_t, beta=0.005, reduction="none") * q_valid
        # Memory autocalls (M > 1) carry a W1 path-covariance gap vs scalar q_mc; relax raw q_mc weight
        # while supervising pv_coupon_pred directly against V3 coupon_pv_mc.
        is_mem = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)] > 0.5
        exact_q_mask = (~is_mem) | (m_count <= 1.5)
        q_sample_w = torch.where(
            exact_q_mask,
            torch.ones_like(m_count),
            torch.full_like(m_count, 0.02),
        )
        p_term = (p_err_sq.sum(dim=1) / m_count) + 0.5 * (p_huber.sum(dim=1) / m_count)
        q_term = q_sample_w * ((q_err_sq.sum(dim=1) / q_count) + 0.5 * (q_huber.sum(dim=1) / q_count))
        prob_mse = (p_term + q_term).mean()
        put_loss = (
            160.0 * F.smooth_l1_loss(l_put_pred, l_t, beta=0.0002)
            + 40.0 * F.mse_loss(l_put_pred, l_t)
        )
        if pv_coupon_pred is not None and pv_coupon_target is not None:
            cpn_t = pv_coupon_target.to(dtype=dtype)
            valid_cpn = (
                (coupon_pv_valid > 0.5)
                if coupon_pv_valid is not None
                else torch.ones_like(pv_coupon_pred, dtype=torch.bool)
            )
            if bool(valid_cpn.any()):
                cpn_loss = (
                    160.0 * F.smooth_l1_loss(pv_coupon_pred[valid_cpn], cpn_t[valid_cpn], beta=0.0002)
                    + 40.0 * F.mse_loss(pv_coupon_pred[valid_cpn], cpn_t[valid_cpn])
                )
            else:
                cpn_loss = torch.zeros((), dtype=dtype, device=pv_pred.device)
        else:
            cpn_loss = torch.zeros((), dtype=dtype, device=pv_pred.device)
        comp_loss = 80.0 * prob_mse + put_loss + cpn_loss

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
        compute_mono: bool = True,
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
        pv_pred, _, pv_coupon_pred, _ = assemble_surrogate_components(
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
            g_node_raw, g_edge_raw, g_sched_raw, g_glob_raw = torch.autograd.grad(
                pv_pred.sum(),
                (node_in, edge_in, sched_in, glob_in),
                create_graph=create_graph_for_sobolev,
                retain_graph=True,
                allow_unused=True,
            )
            g_node = torch.zeros_like(node_in) if g_node_raw is None else g_node_raw
            g_edge = torch.zeros_like(edge_in) if g_edge_raw is None else g_edge_raw
            g_sched = torch.zeros_like(sched_in) if g_sched_raw is None else g_sched_raw
            g_glob = torch.zeros_like(glob_in) if g_glob_raw is None else g_glob_raw
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
        if compute_mono and self.lambda_mono > 0.0 and self.ko_bump_delta > 0.0:
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
            pv_coupon_pred=pv_coupon_pred,
            pv_coupon_target=batch.coupon_pv_mc,
            coupon_pv_valid=batch.coupon_pv_valid,
        )
        # Light shrinkage keeps residual logits calibrated under extreme barrier extrapolation.
        m_cnt = (smask_in > 0.5).to(dtype=pv_pred.dtype).sum(dim=1).clamp(min=1.0)
        logit_reg = 1e-4 * (
            (((surr_out.delta_hazard_logits ** 2) + (surr_out.delta_coupon_logits ** 2)) * smask_in)
            .sum(dim=1)
            / m_cnt
            + (surr_out.delta_put_loss ** 2)
        ).mean()
        breakdown.total_loss = breakdown.total_loss + logit_reg
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


def _compute_difficulty_basis_8d(
    batch: SyntheticAutocallBatch,
    residual_pv_abs: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """Computes the 8D physical difficulty basis phi(x) >= 0 in float64:
    [1.0, sqrt(h_disp * v_bar), ((M_rem - 1)/47) * v_bar, I_WOF * h_disp, I_live_KI * G_put * v_last,
     I_mem * c_mem_0, G_put * L_put_proxy * max(0, 1 - min_m0), |PV_surr - PV_proxy|].
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
    if batch.sched_feats.shape[1] > 0:
        last_idx = (
            (m_counts.long() - 1)
            .clamp(min=0, max=int(batch.sched_feats.shape[1]) - 1)
            .unsqueeze(1)
        )
        v_last = (
            batch.sched_feats[:, :, int(AutocallSchedCol.STULZ_VOL_W)]
            .to(torch.float64)
            .clamp(min=0.0)
            .gather(1, last_idx)
            .squeeze(1)
        )
        v_last = torch.where(m_counts > 0.5, v_last, torch.zeros_like(v_last))
    else:
        v_last = torch.zeros((n_b,), dtype=torch.float64, device=h_disp.device)
    phi_4 = (has_ki & not_ki_past).to(dtype=torch.float64) * g_put * v_last

    is_mem = (batch.global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].to(torch.float64) > 0.5).to(
        dtype=torch.float64
    )
    c_mem_0 = batch.global_feats[:, int(AutocallGlobalCol.C_MEM_0)].to(torch.float64).clamp(min=0.0)
    phi_5 = is_mem * c_mem_0

    l_put_px = batch.global_feats[:, int(AutocallGlobalCol.L_PUT_PROXY)].to(torch.float64).clamp(min=0.0)
    log_m = batch.node_feats[:, :, int(AutocallNodeCol.LOG_MONEYNESS)].to(torch.float64)
    active_bool = batch.asset_mask > 0.5
    m0_vals = torch.exp(log_m.clamp(-10.0, 10.0))
    m0_masked = torch.where(active_bool, m0_vals, torch.full_like(m0_vals, 1e6))
    has_any_a = active_bool.any(dim=1)
    min_m0 = torch.where(has_any_a, m0_masked.min(dim=1).values, torch.ones_like(phi_0))
    phi_6 = g_put * l_put_px * (1.0 - min_m0).clamp(min=0.0)

    if residual_pv_abs is not None:
        res_arr = np.asarray(residual_pv_abs, dtype=np.float64).reshape(-1)
        if res_arr.shape[0] != n_b:
            raise ValueError(f"residual_pv_abs length {res_arr.shape[0]} != batch size {n_b}.")
        phi_7 = torch.as_tensor(np.maximum(res_arr, 0.0), dtype=torch.float64, device=h_disp.device)
    else:
        phi_7 = torch.zeros((n_b,), dtype=torch.float64, device=h_disp.device)

    return (
        torch.stack([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7], dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )


def _compute_difficulty_basis_5d(batch: SyntheticAutocallBatch) -> npt.NDArray[np.float64]:
    """Computes the leading 5D physical difficulty basis phi(x) >= 0 in float64."""
    return _compute_difficulty_basis_8d(batch)[:, :5]


def _compute_local_scale_factor(batch: SyntheticAutocallBatch) -> npt.NDArray[np.float64]:
    """Computes sqrt(h_disp * v_bar) local heteroscedastic scale factor per trade."""
    return _compute_difficulty_basis_8d(batch)[:, 1]


def _extract_mondrian_cell_keys_from_batch(batch: SyntheticAutocallBatch) -> List[str]:
    """Maps each trade in `batch` to its 12-cell Mondrian taxonomy key `ki{0,1}_m{0,1,2}_d{0,1}`."""
    has_ki = (
        (batch.global_feats[:, int(AutocallGlobalCol.HAS_KI_FLAG)] > 0.5)
        & (batch.global_feats[:, int(AutocallGlobalCol.I_KI_PAST)] < 0.5)
    ).detach().cpu().numpy()
    m_cnt = (batch.sched_mask > 0.5).sum(dim=1).detach().cpu().numpy()
    d_cnt = (batch.asset_mask > 0.5).sum(dim=1).detach().cpu().numpy()
    keys: List[str] = []
    for hk, m_c, d_c in zip(has_ki, m_cnt, d_cnt):
        ki_b = 1 if bool(hk) else 0
        m_b = 0 if int(m_c) <= 12 else (1 if int(m_c) <= 25 else 2)
        d_b = 0 if int(d_c) <= 2 else 1
        keys.append(f"ki{ki_b}_m{m_b}_d{d_b}")
    return keys


def _extract_mondrian_cell_keys_24_from_batch(batch: SyntheticAutocallBatch) -> List[str]:
    """Maps each trade in `batch` to its 24-cell Mondrian key `ki{0,1}_m{0,1,2}_d{0,1}_mem{0,1}`."""
    base_keys = _extract_mondrian_cell_keys_from_batch(batch)
    is_mem = (batch.global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)] > 0.5).detach().cpu().numpy()
    return [f"{bk}_mem{1 if bool(mem_f) else 0}" for bk, mem_f in zip(base_keys, is_mem)]


def compute_conformal_half_width_bps(
    batch: SyntheticAutocallBatch,
    stats_payload: Dict[str, Any],
    *,
    pv_pred: Optional[torch.Tensor | npt.NDArray[np.float64]] = None,
) -> npt.NDArray[np.float64]:
    """Computes split/Mondrian-conformal worst-case error bound in bps (8D signed EB-Mondrian or 5D fallback)."""
    if (
        "scale_weights_8d" in stats_payload
        and stats_payload["scale_weights_8d"] is not None
        and "mondrian_q_lo_table" in stats_payload
        and "mondrian_q_hi_table" in stats_payload
    ):
        w8 = np.asarray(
            stats_payload["scale_weights_8d"].detach().cpu().numpy()
            if isinstance(stats_payload["scale_weights_8d"], torch.Tensor)
            else stats_payload["scale_weights_8d"],
            dtype=np.float64,
        ).reshape(8)
        res_abs: Optional[npt.NDArray[np.float64]] = None
        if pv_pred is not None:
            pv_p_np = (
                pv_pred.detach().cpu().numpy().astype(np.float64)
                if isinstance(pv_pred, torch.Tensor)
                else np.asarray(pv_pred, dtype=np.float64)
            )
            pv_px_np = batch.pv_proxy.detach().cpu().numpy().astype(np.float64)
            res_abs = np.abs(pv_p_np - pv_px_np)
        phi_8 = _compute_difficulty_basis_8d(batch, residual_pv_abs=res_abs)
        scale_8 = np.maximum(phi_8 @ w8, 1.5e-4)

        b_glob = float(stats_payload.get("b_global_conformal", 0.0))
        q_lo_glob = float(stats_payload.get("q_lo_conformal", stats_payload.get("q_err_conformal", 0.0)))
        q_hi_glob = float(stats_payload.get("q_hi_conformal", stats_payload.get("q_err_conformal", 0.0)))
        b_tbl = stats_payload.get("mondrian_bias_table") or {}
        q_lo_tbl = stats_payload.get("mondrian_q_lo_table") or {}
        q_hi_tbl = stats_payload.get("mondrian_q_hi_table") or {}
        cell_keys_24 = _extract_mondrian_cell_keys_24_from_batch(batch)
        b_vec = np.array([float(b_tbl.get(k, b_glob)) for k in cell_keys_24], dtype=np.float64)
        q_lo_vec = np.array([float(q_lo_tbl.get(k, q_lo_glob)) for k in cell_keys_24], dtype=np.float64)
        q_hi_vec = np.array([float(q_hi_tbl.get(k, q_hi_glob)) for k in cell_keys_24], dtype=np.float64)
        lo_bound = np.abs(b_vec - q_lo_vec * scale_8)
        hi_bound = np.abs(b_vec + q_hi_vec * scale_8)
        return np.asarray(np.maximum(lo_bound, hi_bound) * 1e4, dtype=np.float64)

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
    mondrian_tbl = stats_payload.get("mondrian_q_err_table")
    if isinstance(mondrian_tbl, dict) and len(mondrian_tbl) > 0:
        cell_keys = _extract_mondrian_cell_keys_from_batch(batch)
        q_vec = np.array([float(mondrian_tbl.get(k, q_err)) for k in cell_keys], dtype=np.float64)
        return np.asarray(q_vec * scale * 1e4, dtype=np.float64)
    return np.asarray(q_err * scale * 1e4, dtype=np.float64)


def _predict_batch_pv(
    model: MaskedEQAutocallSurrogate,
    batch: SyntheticAutocallBatch,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Evaluates predicted ProductValue on CPU in float64 across mini-batches."""
    n = int(batch.node_feats.shape[0])
    if n == 0:
        return torch.empty((0,), dtype=torch.float64)
    was_training = model.training
    model.eval()
    preds: List[torch.Tensor] = []
    try:
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
    finally:
        if was_training:
            model.train()
    return torch.cat(preds, dim=0)


def _fit_quantile_scale_weights_8d(
    phi_fit: npt.NDArray[np.float64],
    err_fit: npt.NDArray[np.float64],
    *,
    tau: float = 0.90,
    l2_reg: float = 1e-4,
    w1_floor: float = 5e-4,
) -> npt.NDArray[np.float64]:
    """Fits L2-regularized non-negative tau-quantile pinball weights w in R>=0^8 with w[1] >= w1_floor."""
    n1, d_phi = phi_fit.shape
    w_init = np.zeros(d_phi, dtype=np.float64)
    w_init[0] = 1.5e-4
    if d_phi > 1:
        w_init[1] = float(w1_floor)

    if n1 >= 32:
        c = np.zeros(d_phi + 2 * n1, dtype=np.float64)
        c[0] = 1e-3
        c[1:d_phi] = 1e-4
        c[d_phi : d_phi + n1] = float(tau) / float(n1)
        c[d_phi + n1 : d_phi + 2 * n1] = (1.0 - float(tau)) / float(n1)
        a_eq = sp_sparse.hstack(
            [
                sp_sparse.csr_matrix(phi_fit),
                sp_sparse.eye(n1, format="csr", dtype=np.float64),
                -sp_sparse.eye(n1, format="csr", dtype=np.float64),
            ],
            format="csr",
        )
        bounds_lp = (
            [(1.5e-4, 3.0e-4), (float(w1_floor), None)]
            + [(0.0, None)] * (d_phi - 2)
            + [(0.0, None)] * (2 * n1)
        )
        res = linprog(c=c, A_eq=a_eq, b_eq=err_fit, bounds=bounds_lp, method="highs")
        if bool(res.success) and res.x is not None:
            w_init = np.asarray(res.x[:d_phi], dtype=np.float64)
        else:
            w_nnls, _ = nnls(phi_fit, err_fit)
            w_init = np.asarray(w_nnls[:d_phi], dtype=np.float64)
    else:
        w_nnls, _ = nnls(phi_fit, err_fit)
        w_init = np.asarray(w_nnls[:d_phi], dtype=np.float64)

    w_init[0] = min(max(float(w_init[0]), 1.5e-4), 3.0e-4)
    if d_phi > 1:
        w_init[1] = max(float(w_init[1]), float(w1_floor))
        w_init[2:] = np.maximum(w_init[2:], 0.0)

    eps_smooth = 1e-5
    inv_n = 1.0 / float(max(n1, 1))
    tau_f = float(tau)

    def _smooth_pinball_obj(w_vec: npt.NDArray[np.float64]) -> Tuple[float, npt.NDArray[np.float64]]:
        r = err_fit - phi_fit @ w_vec
        z = np.clip(r / eps_smooth, -40.0, 40.0)
        sp_pos = np.where(r / eps_smooth > 40.0, r, eps_smooth * np.log1p(np.exp(z)))
        sp_neg = np.where(-r / eps_smooth > 40.0, -r, eps_smooth * np.log1p(np.exp(-z)))
        sig_pos = 1.0 / (1.0 + np.exp(-z))
        sig_neg = 1.0 - sig_pos
        loss = inv_n * float(np.sum(tau_f * sp_pos + (1.0 - tau_f) * sp_neg))
        reg = 0.5 * float(l2_reg) * float(np.sum(w_vec[1:] * w_vec[1:]))
        d_loss_dr = inv_n * (tau_f * sig_pos - (1.0 - tau_f) * sig_neg)
        grad = -(phi_fit.T @ d_loss_dr)
        grad[1:] += float(l2_reg) * w_vec[1:]
        return loss + reg, grad

    bounds_lbfgs = [(1.5e-4, 3.0e-4), (float(w1_floor), None)] + [(0.0, None)] * max(d_phi - 2, 0)
    opt = minimize(
        _smooth_pinball_obj,
        w_init,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds_lbfgs[:d_phi],
    )
    w_out = np.asarray(opt.x if bool(opt.success) else w_init, dtype=np.float64)
    w_out[0] = min(max(float(w_out[0]), 1.5e-4), 3.0e-4)
    if d_phi > 1:
        w_out[1] = max(float(w_out[1]), float(w1_floor))
        w_out[2:] = np.maximum(w_out[2:], 0.0)
    return w_out


def calibrate_guardian_stats(
    model: MaskedEQAutocallSurrogate,
    train_batch: SyntheticAutocallBatch,
    cal_batch: SyntheticAutocallBatch,
    *,
    device: torch.device,
    conformal_alpha: float = 0.05,
    tau_desk_bps: float = 15.0,
    stats_path: Optional[Path] = DEFAULT_GUARDIAN_STATS_PATH,
) -> Dict[str, Any]:
    """Fits OAS covariance, Laplace-smoothed 10-decile PSI bins, and signed 8D EB-Mondrian CQR quantiles."""
    x_fit = extract_16d_features_from_batch(train_batch)
    n_fit, p = x_fit.shape
    ref_mean = np.mean(x_fit, axis=0)
    centered = x_fit - ref_mean[None, :]
    cov_sample = (centered.T @ centered) / float(max(n_fit - 1, 1))
    diag_var = np.maximum(np.diag(cov_sample), 1e-12)
    raw_std = np.sqrt(diag_var)
    marginal_std = np.maximum(raw_std, 1e-2)
    r_sample = cov_sample / np.outer(raw_std, raw_std)
    r_sample = 0.5 * (r_sample + r_sample.T)
    np.fill_diagonal(r_sample, 1.0)
    tr_r = float(np.trace(r_sample))
    tr_r2 = float(np.sum(r_sample * r_sample))
    denom = float(n_fit + 1 - 2.0 / p) * max(tr_r2 - (tr_r * tr_r) / p, 1e-12)
    numer = (1.0 - 2.0 / p) * tr_r2 + tr_r * tr_r
    rho_oas = min(1.0, max(0.05, numer / max(denom, 1e-12)))
    f_target = (tr_r / p) * np.eye(p, dtype=np.float64)
    r_oas = (1.0 - rho_oas) * r_sample + rho_oas * f_target
    r_oas = 0.5 * (r_oas + r_oas.T)
    np.fill_diagonal(r_oas, 1.0)
    cov_oas_in = (marginal_std[:, None] * r_oas) * marginal_std[None, :]
    cov_oas, cov_inv, marginal_std, cov_inv_d1 = _regularize_cov_oas_correlation_space(cov_oas_in)

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
        probs = (counts.astype(np.float64) + 0.5) / (float(max(n_fit, 1)) + 5.0)
        psi_ref_probs[j] = probs / np.sum(probs)

    # Split-conformal calibration strictly on held-out Cal split (Cal-1 scale/bias fit, Cal-2 signed conformal quantiles)
    x_cal = extract_16d_features_from_batch(cal_batch)
    n_cal = int(x_cal.shape[0])
    if n_cal < 2048:
        _LOGGER.warning(
            "Calibrating Guardian conformal quantiles on n_cal=%d (< 2048 recommended for production stability).",
            n_cal,
        )
    diff_cal = x_cal - ref_mean[None, :]
    d2_cal = np.einsum("ni,ij,nj->n", diff_cal, cov_inv, diff_cal)
    q_ood_level = min(1.0, math.ceil((n_cal + 1) * 0.999) / float(max(n_cal, 1)))
    q_ood_conformal = float(np.quantile(d2_cal, q_ood_level))

    pv_pred_cal = _predict_batch_pv(model, cal_batch, device=device).numpy()
    pv_mc_cal = cal_batch.pv_mc.detach().cpu().numpy().astype(np.float64)
    pv_px_cal = cal_batch.pv_proxy.detach().cpu().numpy().astype(np.float64)
    signed_res_cal = pv_mc_cal - pv_pred_cal
    abs_err_cal = np.abs(signed_res_cal)
    res_px_cal = np.abs(pv_pred_cal - pv_px_cal)
    phi_cal_8d = _compute_difficulty_basis_8d(cal_batch, residual_pv_abs=res_px_cal)
    all_cell_keys_12 = _extract_mondrian_cell_keys_from_batch(cal_batch)
    all_cell_keys_24 = _extract_mondrian_cell_keys_24_from_batch(cal_batch)

    tau_q = max(0.85, min(0.98, 1.0 - float(conformal_alpha)))
    m0_eb = 50.0
    if n_cal >= 64:
        split_idx = n_cal // 2
        w8 = _fit_quantile_scale_weights_8d(phi_cal_8d[:split_idx], abs_err_cal[:split_idx], tau=tau_q)
        fit_res = signed_res_cal[:split_idx]
        fit_keys_24 = np.asarray(all_cell_keys_24[:split_idx])
        phi_score = phi_cal_8d[split_idx:]
        signed_score = signed_res_cal[split_idx:]
        err_score = abs_err_cal[split_idx:]
        score_keys_12 = all_cell_keys_12[split_idx:]
        score_keys_24 = all_cell_keys_24[split_idx:]
    else:
        w8 = _fit_quantile_scale_weights_8d(phi_cal_8d, abs_err_cal, tau=tau_q)
        fit_res = signed_res_cal
        fit_keys_24 = np.asarray(all_cell_keys_24)
        phi_score = phi_cal_8d
        signed_score = signed_res_cal
        err_score = abs_err_cal
        score_keys_12 = all_cell_keys_12
        score_keys_24 = all_cell_keys_24

    b_global = float(np.mean(fit_res)) if fit_res.size > 0 else 0.0
    mondrian_bias_table: Dict[str, float] = {}
    for ki_b in (0, 1):
        for m_b in (0, 1, 2):
            for d_b in (0, 1):
                for mem_b in (0, 1):
                    ck24 = f"ki{ki_b}_m{m_b}_d{d_b}_mem{mem_b}"
                    c_mask_f = fit_keys_24 == ck24
                    n_cf = int(np.sum(c_mask_f))
                    if n_cf > 0:
                        b_emp = float(np.mean(fit_res[c_mask_f]))
                        b_hat = (n_cf / (n_cf + m0_eb)) * b_emp + (m0_eb / (n_cf + m0_eb)) * b_global
                    else:
                        b_hat = b_global
                    mondrian_bias_table[ck24] = float(b_hat)

    eff_alpha = float(conformal_alpha) * 0.88 if 0.0 < float(conformal_alpha) < 0.25 else float(conformal_alpha)
    n_score = int(err_score.shape[0])
    denom_score = np.maximum(phi_score @ w8, 1.5e-4)

    # Centered signed standardized residuals on Cal-2 for asymmetric EB-Mondrian quantiles
    b_hat_score = np.array([mondrian_bias_table.get(k, b_global) for k in score_keys_24], dtype=np.float64)
    u_score = (signed_score - b_hat_score) / denom_score
    alpha_side = 0.5 * eff_alpha
    lvl_side = min(1.0, math.ceil((n_score + 1) * (1.0 - alpha_side)) / float(max(n_score, 1)))
    q_hi_global = max(0.0, float(np.quantile(u_score, lvl_side))) if n_score > 0 else 0.0
    q_lo_global = max(0.0, float(np.quantile(-u_score, lvl_side))) if n_score > 0 else 0.0

    mondrian_q_lo_table: Dict[str, float] = {}
    mondrian_q_hi_table: Dict[str, float] = {}
    score_keys_24_arr = np.asarray(score_keys_24)
    for ki_b in (0, 1):
        for m_b in (0, 1, 2):
            for d_b in (0, 1):
                for mem_b in (0, 1):
                    ck24 = f"ki{ki_b}_m{m_b}_d{d_b}_mem{mem_b}"
                    c_mask = score_keys_24_arr == ck24
                    n_c = int(np.sum(c_mask))
                    if n_c > 0:
                        c_u = u_score[c_mask]
                        lvl_c = min(1.0, math.ceil((n_c + 1) * (1.0 - alpha_side)) / float(n_c))
                        q_hi_emp = max(0.0, float(np.quantile(c_u, lvl_c)))
                        q_lo_emp = max(0.0, float(np.quantile(-c_u, lvl_c)))
                        q_hi_c = (n_c / (n_c + m0_eb)) * q_hi_emp + (m0_eb / (n_c + m0_eb)) * q_hi_global
                        q_lo_c = (n_c / (n_c + m0_eb)) * q_lo_emp + (m0_eb / (n_c + m0_eb)) * q_lo_global
                    else:
                        q_hi_c = q_hi_global
                        q_lo_c = q_lo_global
                    mondrian_q_hi_table[ck24] = float(q_hi_c)
                    mondrian_q_lo_table[ck24] = float(q_lo_c)

    # Legacy 12-cell unsigned summary kept populated for backward compatibility
    scores_cal = err_score / denom_score
    q_err_level = min(1.0, math.ceil((n_score + 1) * (1.0 - eff_alpha)) / float(max(n_score, 1)))
    q_err_conformal = float(np.quantile(scores_cal, q_err_level))
    w5 = w8[:5].copy()
    scale_eps0 = float(w8[0])
    scale_c1 = float(w8[1])

    mondrian_q_err_table: Dict[str, float] = {}
    score_keys_12_arr = np.asarray(score_keys_12)
    for ki_b in (0, 1):
        for m_b in (0, 1, 2):
            for d_b in (0, 1):
                ck = f"ki{ki_b}_m{m_b}_d{d_b}"
                c_mask = score_keys_12_arr == ck
                n_c = int(np.sum(c_mask))
                if n_c > 0:
                    c_scores = scores_cal[c_mask]
                    lvl_c = min(1.0, math.ceil((n_c + 1) * (1.0 - eff_alpha)) / float(n_c))
                    q_emp_c = float(np.quantile(c_scores, lvl_c))
                    q_eff_c = (n_c / (n_c + m0_eb)) * q_emp_c + (m0_eb / (n_c + m0_eb)) * q_err_conformal
                else:
                    q_eff_c = q_err_conformal
                mondrian_q_err_table[ck] = float(q_eff_c)

    payload: Dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "conformal_mode": "signed_8d_eb_mondrian",
        "feature_names_16d": list(FEATURE_NAMES_16D),
        "ref_mean_16d": torch.from_numpy(ref_mean).to(torch.float64),
        "ref_cov_oas_16d": torch.from_numpy(cov_oas_in).to(torch.float64),
        "ref_cov_inv_16d": torch.from_numpy(cov_inv).to(torch.float64),
        "ref_cov_inv_d1_16d": torch.from_numpy(cov_inv_d1).to(torch.float64),
        "ref_marginal_std_16d": torch.from_numpy(marginal_std).to(torch.float64),
        "psi_bin_edges": torch.from_numpy(psi_bin_edges).to(torch.float64),
        "psi_ref_probs": torch.from_numpy(psi_ref_probs).to(torch.float64),
        "conformal_alpha": float(conformal_alpha),
        "tau_desk_bps": float(tau_desk_bps),
        "q_ood_conformal": q_ood_conformal,
        "q_err_conformal": q_err_conformal,
        "b_global_conformal": b_global,
        "q_lo_conformal": q_lo_global,
        "q_hi_conformal": q_hi_global,
        "scale_weights_5d": torch.from_numpy(w5).to(torch.float64),
        "scale_weights_8d": torch.from_numpy(w8).to(torch.float64),
        "scale_eps0": scale_eps0,
        "scale_c1": scale_c1,
        "n_cal": n_cal,
        "mondrian_q_err_table": mondrian_q_err_table,
        "mondrian_bias_table": mondrian_bias_table,
        "mondrian_q_lo_table": mondrian_q_lo_table,
        "mondrian_q_hi_table": mondrian_q_hi_table,
        "oas_standardized": True,
    }
    if stats_path is not None:
        _atomic_torch_save(payload, Path(stats_path))
    return payload


def evaluate_conformal_coverage(
    model: MaskedEQAutocallSurrogate,
    test_batch: SyntheticAutocallBatch,
    stats_payload: Dict[str, Any],
    *,
    device: torch.device,
) -> float:
    """Computes empirical split-conformal coverage on a held-out test batch."""
    pv_pred = _predict_batch_pv(model, test_batch, device=device)
    pv_mc = test_batch.pv_mc.detach().cpu().numpy().astype(np.float64)
    abs_err_bps = np.abs(pv_pred.numpy() - pv_mc) * 1e4
    half_width_bps = compute_conformal_half_width_bps(test_batch, stats_payload, pv_pred=pv_pred)
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
        if pv_mc.numel() == 0:
            return 0.0, 0.0, {}
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
    n = int(batch.node_feats.shape[0])
    if n == 0:
        return 0.0, 0.0
    was_training = model.training
    model.eval()
    tot_pred_list: List[torch.Tensor] = []
    tot_tgt_list: List[torch.Tensor] = []
    res_pred_list: List[torch.Tensor] = []
    res_tgt_list: List[torch.Tensor] = []

    try:
        with torch.enable_grad():  # type: ignore[no-untyped-call]
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
    finally:
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
    n_cal_samples: int = 4096,
    n_test_samples: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 80,
    lr: float = 2.2e-3,
    weight_decay: float = 2.0e-4,
    n_mc_paths_train: int = 12288,
    n_substeps_train: int = 8,
    n_mc_paths_test: int = 32768,
    n_substeps_test: int = 16,
    d_max: int = D_MAX_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
    d_model: int = 128,
    n_egno_layers: int = 3,
    n_sched_layers: int = 3,
    n_heads: int = 4,
    n_func_codes: int = 18,
    func_emb_dim: int = 16,
    seed: int = 2026,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    guardian_stats_path: str | Path | None = DEFAULT_GUARDIAN_STATS_PATH,
    conformal_alpha: float = 0.05,
    tau_desk_bps: float = 15.0,
    domain_mode: str = "tier2_desk",
    cv_mode_test: str = "multifidelity_gbm",
    data_dir: str | Path | None = None,
) -> SurrogateTrainingResult:
    """Trains EGNO v2.0 on V3 GPU Dupire LV QMC labels and calibrates signed 8D EB-Mondrian Guardian stats."""
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

    def _load_or_gen_split(
        split_name: str,
        gen_fn: Callable[[], SyntheticAutocallBatch],
        *,
        expected_samples: int,
        expected_domain_mode: str,
        expected_seed: int,
    ) -> SyntheticAutocallBatch:
        return _load_or_gen_cached_split(
            split_name,
            gen_fn,
            data_dir=data_dir,
            n_samples=expected_samples,
            domain_mode=expected_domain_mode,
            seed=expected_seed,
            d_max=d_max,
            m_max=m_max,
            save_on_miss=True,
        )

    eff_cv_test = cv_mode_test if n_mc_paths_test >= 4096 else "none"

    # Mix 85% tier2_desk + 15% full_stress for training while keeping OAS calibration purely tier2_desk.
    if domain_mode == "tier2_desk" and n_train_samples >= 128:
        n_desk_train = int(round(0.85 * n_train_samples))
        n_stress_train = max(0, n_train_samples - n_desk_train)
        train_desk = _load_or_gen_split(
            "train_desk",
            lambda: generate_synthetic_autocall_batch(
                batch_size=n_desk_train,
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
                domain_mode="tier2_desk",
            ),
            expected_samples=n_desk_train,
            expected_domain_mode="tier2_desk",
            expected_seed=seed,
        )
        if n_stress_train > 0:
            train_stress = _load_or_gen_split(
                "train_stress",
                lambda: generate_synthetic_autocall_batch(
                    batch_size=n_stress_train,
                    d_max=d_max,
                    m_max=m_max,
                    m_rem_max=m_max,
                    n_mc_paths=n_mc_paths_train,
                    n_substeps=n_substeps_train,
                    seed=seed + 50_000,
                    device="cpu",
                    qmc_device=dev,
                    dtype=torch.float32,
                    cv_mode="none",
                    domain_mode="full_stress",
                ),
                expected_samples=n_stress_train,
                expected_domain_mode="full_stress",
                expected_seed=seed + 50_000,
            )
            train_data = _concat_batches([train_desk, train_stress])
        else:
            train_data = train_desk
    else:
        train_desk = _load_or_gen_split(
            "train_desk",
            lambda: generate_synthetic_autocall_batch(
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
            ),
            expected_samples=n_train_samples,
            expected_domain_mode=domain_mode,
            expected_seed=seed,
        )
        train_data = train_desk
    val_data = _load_or_gen_split(
        "val",
        lambda: generate_synthetic_autocall_batch(
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
            compute_pathwise_delta=False,
            domain_mode=domain_mode,
        ),
        expected_samples=n_val_samples,
        expected_domain_mode=domain_mode,
        expected_seed=seed + 100_000,
    )
    cal_data = _load_or_gen_split(
        "cal",
        lambda: generate_synthetic_autocall_batch(
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
            cv_mode=eff_cv_test,
            compute_pathwise_delta=False,
            domain_mode=domain_mode,
        ),
        expected_samples=n_cal_samples,
        expected_domain_mode=domain_mode,
        expected_seed=seed + 200_000,
    )
    test_data = _load_or_gen_split(
        "test",
        lambda: generate_synthetic_autocall_batch(
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
            cv_mode=eff_cv_test,
            domain_mode=domain_mode,
        ),
        expected_samples=n_test_samples,
        expected_domain_mode=domain_mode,
        expected_seed=seed + 300_000,
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

    # 2. Two-stage EGNO v2.0 optimization: Stage 1 fast non-Sobolev warmup, Stage 2 Sobolev H^1 + mono fine-tune.
    criterion = SpreadAttenuatedSobolevLoss().to(device=dev)
    steps_per_epoch = max(1, math.ceil(n_train_samples / batch_size))
    n_stage1 = int(n_epochs * 0.65) if n_epochs >= 12 else 0

    if n_stage1 > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        s1_steps = max(4, n_stage1 * steps_per_epoch)
        warmup_steps = min(max(1, s1_steps // 6), 3 * steps_per_epoch)

        def _s1_lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 0.1 + 0.9 * float(step + 1) / float(warmup_steps)
            denom = max(1, s1_steps - warmup_steps)
            prog = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
            return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_s1_lr_lambda
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(4, n_epochs * steps_per_epoch),
            eta_min=1e-5,
        )

    loss_history: List[float] = []
    best_stage1_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_stage1_val = float("inf")
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_mae_bps = float("inf")
    last_cos_sim = 0.0

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 99)

    for _epoch in range(n_epochs):
        if _epoch == n_stage1 and n_stage1 > 0:
            model.load_state_dict(
                {k: v.to(device=dev) for k, v in best_stage1_state_dict.items()}, strict=True
            )
            best_val_mae_bps = float("inf")
            s2_epochs = max(1, n_epochs - n_stage1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.35, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(4, s2_epochs * steps_per_epoch),
                eta_min=1e-5,
            )
        use_sobolev = _epoch >= n_stage1
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
                compute_mono=use_sobolev,
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
        if not use_sobolev:
            if val_mae_bps < best_stage1_val:
                best_stage1_val = val_mae_bps
                best_stage1_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_mae_bps = val_mae_bps
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif val_mae_bps < best_val_mae_bps:
            best_val_mae_bps = val_mae_bps
            last_cos_sim = avg_cos
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (_epoch + 1) % 5 == 0 or _epoch == 0 or (_epoch + 1) == n_epochs or (_epoch + 1) == n_stage1:
            stage_tag = "S2" if use_sobolev else "S1"
            cur_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"[{stage_tag} epoch {_epoch + 1:3d}/{n_epochs}] lr={cur_lr:.2e} loss={avg_loss:.4f} "
                f"val_mae={val_mae_bps:.2f}bps (best={best_val_mae_bps:.2f}bps) "
                f"delta_cos={avg_cos:+.4f} elapsed={time.perf_counter() - t_start:.1f}s",
                flush=True,
            )

    if n_stage1 > 0 and n_stage1 < n_epochs and best_val_mae_bps > best_stage1_val * 1.25:
        best_state_dict = best_stage1_state_dict
        best_val_mae_bps = best_stage1_val
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
    eff_res_cos_sim = oos_res_cos_sim if n_stage1 == 0 and n_epochs < 4 else (
        oos_res_cos_sim if oos_res_cos_sim != 0.0 else last_cos_sim
    )

    # 4. Calibrate standalone SR 26-2 Guardian artifact on Cal split and audit coverage on Test split
    stats_path_resolved = Path(guardian_stats_path) if guardian_stats_path is not None else None
    stats_payload = calibrate_guardian_stats(
        model,
        train_desk,
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
    hw_bps = compute_conformal_half_width_bps(test_data, stats_payload, pv_pred=test_preds)
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
        _atomic_torch_save(payload, ckpt_path_resolved)

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
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict checkpoint payload in {ckpt_path}, got {type(raw).__name__}.")

    schema_ver = raw.get("schema_version")
    if schema_ver is not None and str(schema_ver) not in ("1.0", "2.0", FEATURE_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported surrogate checkpoint schema_version={schema_ver!r} in {ckpt_path}."
        )

    def _infer_arch_from_state_dict(sd: Dict[str, Any]) -> Dict[str, int]:
        np_w = sd.get("node_proj.weight")
        emb_w = sd.get("pool.emb_call.weight")
        rt_w = sd.get("sched_transformer.rel_time_bias.weight")
        egno_indices = [
            int(k.split(".")[1])
            for k in sd
            if k.startswith("egno_layers.") and k.endswith(".norm.weight") and k.split(".")[1].isdigit()
        ]
        sched_indices = [
            int(k.split(".")[2])
            for k in sd
            if k.startswith("sched_transformer.layers.") and k.endswith(".norm1.weight") and k.split(".")[2].isdigit()
        ]
        return {
            "d_model": int(np_w.shape[0]) if isinstance(np_w, torch.Tensor) and np_w.ndim == 2 else 128,
            "d_node": int(np_w.shape[1]) if isinstance(np_w, torch.Tensor) and np_w.ndim == 2 else 16,
            "n_egno_layers": (max(egno_indices) + 1) if egno_indices else 3,
            "n_sched_layers": (max(sched_indices) + 1) if sched_indices else 2,
            "n_heads": int(rt_w.shape[0]) if isinstance(rt_w, torch.Tensor) and rt_w.ndim == 2 else 4,
            "n_func_codes": int(emb_w.shape[0]) if isinstance(emb_w, torch.Tensor) and emb_w.ndim == 2 else 18,
            "func_emb_dim": int(emb_w.shape[1]) if isinstance(emb_w, torch.Tensor) and emb_w.ndim == 2 else 16,
        }

    if "model_state_dict" in raw or "state_dict" in raw:
        state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw["state_dict"]
        cfg_raw = raw.get("config")
        cfg = cfg_raw if isinstance(cfg_raw, dict) else {}
        cfg_schema = cfg.get("schema_version")
        if cfg_schema is not None and str(cfg_schema) not in ("1.0", "2.0", FEATURE_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported config schema_version={cfg_schema!r} in {ckpt_path}."
            )
        inferred = _infer_arch_from_state_dict(state_dict if isinstance(state_dict, dict) else {})
        model = MaskedEQAutocallSurrogate(
            d_max=int(cfg.get("d_max", 8)),
            m_max=int(cfg.get("m_max", 48)),
            d_model=int(cfg.get("d_model", inferred["d_model"])),
            n_egno_layers=int(cfg.get("n_egno_layers", inferred["n_egno_layers"])),
            n_sched_layers=int(cfg.get("n_sched_layers", inferred["n_sched_layers"])),
            n_heads=int(cfg.get("n_heads", inferred["n_heads"])),
            d_node=int(cfg.get("d_node", inferred["d_node"])),
            d_edge=int(cfg.get("d_edge", 4)),
            d_sched=int(cfg.get("d_sched", 12)),
            d_global=int(cfg.get("d_global", 16)),
            n_func_codes=int(cfg.get("n_func_codes", inferred["n_func_codes"])),
            func_emb_dim=int(cfg.get("func_emb_dim", inferred["func_emb_dim"])),
        )
        model.load_state_dict(state_dict, strict=True)
        raw_metrics = raw.get("metrics")
        metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) else {}
    else:
        inferred = _infer_arch_from_state_dict(raw)
        model = MaskedEQAutocallSurrogate(
            d_model=inferred["d_model"],
            n_egno_layers=inferred["n_egno_layers"],
            n_sched_layers=inferred["n_sched_layers"],
            n_heads=inferred["n_heads"],
            d_node=inferred["d_node"],
            n_func_codes=inferred["n_func_codes"],
            func_emb_dim=inferred["func_emb_dim"],
        )
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
    n_cal_samples: int = 4096,
    n_test_samples: int = 2048,
    n_mc_paths_test: int = 32768,
    n_substeps_test: int = 16,
    seed: int = 2026,
    device: torch.device | str | None = None,
    conformal_alpha: float = 0.05,
    tau_desk_bps: float = 15.0,
    domain_mode: str = "tier2_desk",
    cv_mode_test: str = "multifidelity_gbm",
    data_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Refreshes 8D signed EB-Mondrian Guardian artifact & test metrics while keeping EGNO weights."""
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

    eff_cv_test = cv_mode_test if n_mc_paths_test >= 4096 else "none"

    def _load_or_gen_recal(
        split_name: str,
        gen_fn: Callable[[], SyntheticAutocallBatch],
        *,
        expected_samples: int,
        expected_seed: int,
    ) -> SyntheticAutocallBatch:
        return _load_or_gen_cached_split(
            split_name,
            gen_fn,
            data_dir=data_dir,
            n_samples=expected_samples,
            domain_mode=domain_mode,
            seed=expected_seed,
            d_max=d_max,
            m_max=m_max,
            save_on_miss=False,
        )

    # OAS covariance and PSI bins depend strictly on the 16D feature distribution (identical Sobol contracts).
    train_data = _load_or_gen_recal(
        "train_desk",
        lambda: generate_synthetic_autocall_batch(
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
        ),
        expected_samples=n_train_samples,
        expected_seed=seed,
    )
    cal_data = _load_or_gen_recal(
        "cal",
        lambda: generate_synthetic_autocall_batch(
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
            cv_mode=eff_cv_test,
            compute_pathwise_delta=False,
            domain_mode=domain_mode,
        ),
        expected_samples=n_cal_samples,
        expected_seed=seed + 200_000,
    )
    test_data = _load_or_gen_recal(
        "test",
        lambda: generate_synthetic_autocall_batch(
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
            cv_mode=eff_cv_test,
            compute_pathwise_delta=True,
            domain_mode=domain_mode,
        ),
        expected_samples=n_test_samples,
        expected_seed=seed + 300_000,
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
    hw_bps = compute_conformal_half_width_bps(test_data, stats_payload, pv_pred=test_preds)
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
        _atomic_torch_save(raw, ckpt_path)

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
    parser.add_argument("--n-cal", type=int, default=4096)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2.2e-3)
    parser.add_argument("--n-mc-train", type=int, default=12288)
    parser.add_argument("--n-substeps-train", type=int, default=8)
    parser.add_argument("--n-mc-test", type=int, default=32768)
    parser.add_argument("--n-substeps-test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--domain-mode", type=str, default="tier2_desk")
    parser.add_argument("--tau-desk-bps", type=float, default=15.0)
    parser.add_argument("--recalibrate-only", action="store_true")
    parser.add_argument("--cv-mode-test", type=str, default="multifidelity_gbm")
    parser.add_argument("--data-dir", type=str, default=None)
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
            cv_mode_test=args.cv_mode_test,
            data_dir=args.data_dir,
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
        cv_mode_test=args.cv_mode_test,
        data_dir=args.data_dir,
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
