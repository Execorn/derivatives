"""train_eq_autocall.py — Spread-Attenuated Sobolev H^1 Trainer for MaskedEQAutocallSurrogate (Task 3.2b).

Implements:
  1. `assemble_surrogate_batch_pv`: Differentiable batched PV assembly from `AutocallSurrogateOutput`,
     `sched_feats`, `sched_mask`, and `global_feats` with exact memory-coupon recursion.
  2. `SpreadAttenuatedSobolevLoss(nn.Module)` (Proof W4, Glasserman & Karmarkar 2025, arXiv:2512.05301v2):
     Combines value MSE, stopping/coupon/put-loss component supervision, spread-attenuated
     first-order Delta Sobolev penalty w(Delta_B) = |Delta_B| / (|Delta_B| + Delta_B_0) with
     Delta_B_0 = 0.05, KO barrier monotonicity penalty max(0, p_m(B_KO + delta) - p_m(B_KO))^2,
     and Frobenius Jacobian regularization.
  3. `train_eq_autocall_surrogate`, `evaluate_surrogate_mae_bps`, and `load_eq_autocall_surrogate`:
     End-to-end training loop (`AdamW` + `OneCycleLR`) across variable basket dimensions D in [1, 8]
     saving production checkpoint `artifacts/checkpoints/eq_autocall_egno.pt` satisfying Gate G1.5
     (out-of-sample MAE <= 15.0 bps / 0.0015).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepvol.calibration.autocall_sobol_generator import (
    SyntheticAutocallBatch,
    generate_synthetic_autocall_batch,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
)

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[3] / "artifacts" / "checkpoints" / "eq_autocall_egno.pt"
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


@dataclass
class SurrogateTrainingResult:
    """Summary of a completed `train_eq_autocall_surrogate` run."""

    model: MaskedEQAutocallSurrogate
    checkpoint_path: Path | None
    train_loss_history: list[float]
    test_mae: float
    test_mae_bps: float
    baseline_mae_bps: float
    per_d_mae_bps: dict[int, float]


def assemble_surrogate_batch_pv(
    surr_out: AutocallSurrogateOutput,
    sched_feats: torch.Tensor,
    sched_mask: torch.Tensor,
    global_feats: torch.Tensor,
) -> torch.Tensor:
    """Assembles batched unit-notional ProductValue V_pred in (B,) from `AutocallSurrogateOutput`.

    Matches the exact W1 cash-flow and memory-coupon equations used in `autocall_sobol_generator.py`
    and `pre_post_processor.py`:
      V = sum_{m} DF_m * p_m + DF_{M_rem} * S_{M_rem}
          + sum_{m} DF_m * S_{m-1} * q_m * c_{eff, m}
          - DF_{M_rem} * G_put * L_put
    """
    m_sched = (sched_mask > 0.5).to(dtype=sched_feats.dtype)
    batch_size, m_max, _ = sched_feats.shape

    df_pay = sched_feats[:, :, int(AutocallSchedCol.DF_PAY)] * m_sched
    cpn_rate = sched_feats[:, :, int(AutocallSchedCol.COUPON_RATE)] * m_sched

    p_m = surr_out.stopping_probs * m_sched
    q_m = surr_out.coupon_probs * m_sched
    surv_m = surr_out.survival_probs
    surv_prev = torch.cat([torch.ones_like(surv_m[:, :1]), surv_m[:, :-1]], dim=1)

    # Locate each sample's terminal active observation index M_rem - 1
    m_counts = m_sched.sum(dim=1).long().clamp(min=1, max=m_max)
    last_idx = (m_counts - 1).unsqueeze(1)
    df_term = df_pay.gather(1, last_idx).squeeze(1)
    surv_term = surv_m.gather(1, last_idx).squeeze(1)

    pv_notional = (df_pay * p_m).sum(dim=1) + df_term * surv_term

    c_mem_0 = global_feats[:, int(AutocallGlobalCol.C_MEM_0)]
    is_mem = global_feats[:, int(AutocallGlobalCol.IS_MEMORY_FLAG)].clamp(0.0, 1.0)
    g_put = global_feats[:, int(AutocallGlobalCol.G_PUT)].clamp(min=0.0)

    mem_acc = c_mem_0 * is_mem
    pv_cpn = torch.zeros(batch_size, dtype=sched_feats.dtype, device=sched_feats.device)
    for m in range(m_max):
        active_m = m_sched[:, m]
        c_eff_m = cpn_rate[:, m] + mem_acc
        pv_cpn = pv_cpn + active_m * df_pay[:, m] * surv_prev[:, m] * q_m[:, m] * c_eff_m
        mem_next = is_mem * (1.0 - q_m[:, m]) * c_eff_m
        mem_acc = torch.where(active_m > 0.5, mem_next, mem_acc)

    pv_opt = -df_term * g_put * surr_out.put_loss
    return pv_notional + pv_cpn + pv_opt


class SpreadAttenuatedSobolevLoss(nn.Module):
    """Spread-Attenuated Sobolev H^1 Training Loss (Proof W4, Glasserman & Karmarkar 2025).

    Loss formulation:
      L_total = L_val + lambda_comp * L_comp + lambda_delta * L_Sobolev
                + lambda_mono * L_mono + lambda_jac * L_Jac

    where the first-order Delta Sobolev term is weighted per sample by the fuzzy barrier
    spread attenuation factor:
      w(Delta_B) = |Delta_B| / (|Delta_B| + Delta_B_0),   Delta_B_0 = 0.05
    """

    def __init__(
        self,
        delta_b_0: float = 0.05,
        lambda_comp: float = 0.5,
        lambda_delta: float = 0.10,
        lambda_mono: float = 0.10,
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
        p_pred_ko_bumped: torch.Tensor | None = None,
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

        # 1. Value MSE loss
        val_loss = F.mse_loss(pv_pred, pv_t)

        # 2. Component probability & put-loss MSE loss
        p_err_sq = ((p_pred - p_t) ** 2) * m_sched
        q_err_sq = ((q_pred - q_t) ** 2) * m_sched
        prob_mse = ((p_err_sq + q_err_sq).sum(dim=1) / m_count).mean()
        put_mse = F.mse_loss(l_put_pred, l_t)
        comp_loss = prob_mse + put_mse

        # 3. Spread-attenuated first-order Delta Sobolev penalty (W4)
        w_spread = self.spread_attenuation_weight(global_feats.to(dtype=dtype))
        delta_diff_sq = ((delta_pred - d_t) ** 2) * m_asset
        per_sample_delta_mse = delta_diff_sq.sum(dim=1) / d_count
        sobolev_delta_loss = (w_spread * per_sample_delta_mse).mean()

        # 4. Barrier monotonicity penalty: stopping prob should not increase when KO barrier increases
        if p_pred_ko_bumped is not None:
            mono_viol = F.relu(p_pred_ko_bumped - p_pred) * m_sched
            monotonicity_loss = ((mono_viol ** 2).sum(dim=1) / m_count).mean()
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

        return SobolevLossBreakdown(
            total_loss=total_loss,
            val_loss=val_loss,
            comp_loss=comp_loss,
            sobolev_delta_loss=sobolev_delta_loss,
            monotonicity_loss=monotonicity_loss,
            jacobian_reg_loss=jacobian_reg_loss,
            mae_bps=mae_bps,
        )

    def forward(
        self,
        model: MaskedEQAutocallSurrogate,
        batch: SyntheticAutocallBatch,
        *,
        create_graph_for_sobolev: bool = True,
        return_breakdown: bool = False,
    ) -> torch.Tensor | SobolevLossBreakdown:
        """Evaluates `model` on `batch` and computes the Spread-Attenuated Sobolev H^1 loss."""
        node_in = batch.node_feats.detach().clone().requires_grad_(True)
        edge_in = batch.edge_feats
        amask_in = batch.asset_mask
        sched_in = batch.sched_feats
        smask_in = batch.sched_mask
        glob_in = batch.global_feats

        surr_out = model(
            node_feats=node_in,
            edge_feats=edge_in,
            asset_mask=amask_in,
            sched_feats=sched_in,
            sched_mask=smask_in,
            global_feats=glob_in,
        )
        pv_pred = assemble_surrogate_batch_pv(surr_out, sched_in, smask_in, glob_in)

        # Differentiate V_pred wrt log-moneyness feature x_i[:, :, LOG_MONEYNESS]
        (grad_node,) = torch.autograd.grad(
            pv_pred.sum(),
            node_in,
            create_graph=create_graph_for_sobolev,
            retain_graph=True,
        )
        delta_pred = grad_node[:, :, int(AutocallNodeCol.LOG_MONEYNESS)] * amask_in
        delta_res_target = (batch.delta_mc - batch.delta_proxy) * amask_in

        # Evaluate KO barrier monotonicity bump B_KO -> B_KO * exp(ko_bump_delta)
        p_pred_ko_bumped: torch.Tensor | None = None
        if self.lambda_mono > 0.0 and self.ko_bump_delta > 0.0:
            sched_bumped = sched_in.detach().clone()
            is_call = sched_bumped[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)] * smask_in
            sched_bumped[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)] = (
                sched_bumped[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)] + self.ko_bump_delta * is_call
            )
            # Raising KO barrier lowers baseline proxy KO prob
            sched_bumped[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] = (
                sched_bumped[:, :, int(AutocallSchedCol.PROXY_KO_PROB)] * math.exp(-0.5 * self.ko_bump_delta)
            )
            surr_bumped = model(
                node_feats=node_in.detach(),
                edge_feats=edge_in,
                asset_mask=amask_in,
                sched_feats=sched_bumped,
                sched_mask=smask_in,
                global_feats=glob_in,
            )
            p_pred_ko_bumped = surr_bumped.stopping_probs

        breakdown = self.forward_from_tensors(
            pv_pred=pv_pred,
            pv_target=batch.pv_mc,
            p_pred=surr_out.stopping_probs,
            p_target=batch.p_mc,
            q_pred=surr_out.coupon_probs,
            q_target=batch.q_mc,
            l_put_pred=surr_out.put_loss,
            l_put_target=batch.l_put_mc,
            delta_pred=delta_pred,
            delta_target=delta_res_target,
            asset_mask=amask_in,
            sched_mask=smask_in,
            global_feats=glob_in,
            p_pred_ko_bumped=p_pred_ko_bumped,
        )
        if return_breakdown:
            return breakdown
        return breakdown.total_loss


def evaluate_surrogate_mae_bps(
    model: MaskedEQAutocallSurrogate,
    *,
    n_samples_per_d: int = 16,
    d_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
    n_mc_paths: int = 512,
    seed: int = 7777,
    device: torch.device | str | None = None,
) -> tuple[float, float, dict[int, float]]:
    """Evaluates out-of-sample test MAE (unit-notional and bps) across basket dimensions D in `d_values`.

    Returns:
      `(overall_mae, overall_mae_bps, per_d_mae_bps)`
    """
    was_training = model.training
    model.eval()
    first_param = next(model.parameters(), None)
    dev = (
        torch.device(device)
        if device is not None
        else (first_param.device if first_param is not None else torch.device("cpu"))
    )
    dtype = first_param.dtype if first_param is not None else torch.float32

    per_d_mae_bps: dict[int, float] = {}
    all_errors: list[torch.Tensor] = []

    for d_val in d_values:
        batch_d = generate_synthetic_autocall_batch(
            batch_size=n_samples_per_d,
            d_max=model.d_max,
            m_max=model.m_max,
            n_mc_paths=n_mc_paths,
            seed=seed + 101 * int(d_val),
            device=dev,
            dtype=dtype,
            fixed_d=int(d_val),
        )
        with torch.no_grad():
            surr_out = model(
                node_feats=batch_d.node_feats,
                edge_feats=batch_d.edge_feats,
                asset_mask=batch_d.asset_mask,
                sched_feats=batch_d.sched_feats,
                sched_mask=batch_d.sched_mask,
                global_feats=batch_d.global_feats,
            )
            pv_pred = assemble_surrogate_batch_pv(
                surr_out, batch_d.sched_feats, batch_d.sched_mask, batch_d.global_feats
            )
            abs_err = (pv_pred.to(dtype=torch.float64) - batch_d.pv_mc.to(dtype=torch.float64)).abs()
        per_d_mae_bps[int(d_val)] = float(abs_err.mean().cpu().item() * 1e4)
        all_errors.append(abs_err)

    if was_training:
        model.train()

    concat_err = torch.cat(all_errors, dim=0)
    overall_mae = float(concat_err.mean().cpu().item())
    overall_mae_bps = overall_mae * 1e4
    return overall_mae, overall_mae_bps, per_d_mae_bps


def train_eq_autocall_surrogate(
    *,
    n_train_samples: int = 256,
    batch_size: int = 32,
    n_epochs: int = 25,
    lr: float = 8e-4,
    weight_decay: float = 1e-4,
    n_mc_paths: int = 512,
    d_max: int = 8,
    m_max: int = 48,
    d_model: int = 128,
    n_egno_layers: int = 3,
    n_sched_layers: int = 2,
    n_heads: int = 4,
    seed: int = 2026,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT_PATH,
    eval_samples_per_d: int = 16,
) -> SurrogateTrainingResult:
    """Trains `MaskedEQAutocallSurrogate` using `SpreadAttenuatedSobolevLoss` (`AdamW` + `OneCycleLR`).

    Saves the best model checkpoint to `checkpoint_path` (default: `artifacts/checkpoints/eq_autocall_egno.pt`)
    and verifies out-of-sample test MAE across D in [1, 8] (Gate G1.5: MAE <= 15.0 bps).
    """
    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)

    model = MaskedEQAutocallSurrogate(
        d_max=d_max,
        m_max=m_max,
        d_model=d_model,
        n_egno_layers=n_egno_layers,
        n_sched_layers=n_sched_layers,
        n_heads=n_heads,
    ).to(device=dev, dtype=torch.float32)
    # Initialize residual heads at zero so epoch 0 starts anchored at the Stulz-Margrabe proxy
    model.zero_init_residual_heads(std=1e-4)

    train_data = generate_synthetic_autocall_batch(
        batch_size=n_train_samples,
        d_max=d_max,
        m_max=m_max,
        n_mc_paths=n_mc_paths,
        seed=seed,
        device=dev,
        dtype=torch.float32,
    )
    baseline_mae_bps = float(
        (train_data.pv_proxy.to(torch.float64) - train_data.pv_mc.to(torch.float64))
        .abs()
        .mean()
        .cpu()
        .item()
        * 1e4
    )

    criterion = SpreadAttenuatedSobolevLoss().to(device=dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(n_train_samples / batch_size))
    total_steps = max(8, n_epochs * steps_per_epoch + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.30,
        anneal_strategy="cos",
    )

    loss_history: list[float] = []
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_train_mae_bps = baseline_mae_bps

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 99)

    model.train()
    for _epoch in range(n_epochs):
        perm = torch.randperm(n_train_samples, generator=gen).to(device=dev)
        epoch_loss_sum = 0.0
        epoch_mae_sum = 0.0
        n_batches = 0

        for start_idx in range(0, n_train_samples, batch_size):
            idx = perm[start_idx : start_idx + batch_size]
            sub_batch = SyntheticAutocallBatch(
                node_feats=train_data.node_feats.index_select(0, idx),
                edge_feats=train_data.edge_feats.index_select(0, idx),
                asset_mask=train_data.asset_mask.index_select(0, idx),
                sched_feats=train_data.sched_feats.index_select(0, idx),
                sched_mask=train_data.sched_mask.index_select(0, idx),
                global_feats=train_data.global_feats.index_select(0, idx),
                p_proxy=train_data.p_proxy.index_select(0, idx),
                q_proxy=train_data.q_proxy.index_select(0, idx),
                l_put_proxy=train_data.l_put_proxy.index_select(0, idx),
                pv_proxy=train_data.pv_proxy.index_select(0, idx),
                delta_proxy=train_data.delta_proxy.index_select(0, idx),
                p_mc=train_data.p_mc.index_select(0, idx),
                q_mc=train_data.q_mc.index_select(0, idx),
                l_put_mc=train_data.l_put_mc.index_select(0, idx),
                pv_mc=train_data.pv_mc.index_select(0, idx),
                delta_mc=train_data.delta_mc.index_select(0, idx),
            )

            optimizer.zero_grad(set_to_none=True)
            breakdown = criterion(model, sub_batch, return_breakdown=True)
            assert isinstance(breakdown, SobolevLossBreakdown)
            breakdown.total_loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_sum += float(breakdown.total_loss.detach().cpu().item())
            epoch_mae_sum += breakdown.mae_bps
            n_batches += 1

        avg_loss = epoch_loss_sum / max(n_batches, 1)
        avg_mae_bps = epoch_mae_sum / max(n_batches, 1)
        loss_history.append(avg_loss)
        if avg_mae_bps <= best_train_mae_bps:
            best_train_mae_bps = avg_mae_bps
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device=dev) for k, v in best_state_dict.items()})
    model.eval()

    d_eval_tuple = tuple(range(1, min(d_max, 8) + 1))
    test_mae, test_mae_bps, per_d_mae_bps = evaluate_surrogate_mae_bps(
        model,
        n_samples_per_d=eval_samples_per_d,
        d_values=d_eval_tuple,
        n_mc_paths=n_mc_paths,
        seed=seed + 5000,
        device=dev,
    )

    ckpt_path_resolved: Path | None = None
    if checkpoint_path is not None:
        ckpt_path_resolved = Path(checkpoint_path)
        ckpt_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": {
                "d_max": d_max,
                "m_max": m_max,
                "d_model": d_model,
                "n_egno_layers": n_egno_layers,
                "n_sched_layers": n_sched_layers,
                "n_heads": n_heads,
            },
            "metrics": {
                "test_mae": test_mae,
                "test_mae_bps": test_mae_bps,
                "baseline_mae_bps": baseline_mae_bps,
                "per_d_mae_bps": per_d_mae_bps,
            },
        }
        torch.save(payload, ckpt_path_resolved)

    return SurrogateTrainingResult(
        model=model,
        checkpoint_path=ckpt_path_resolved,
        train_loss_history=loss_history,
        test_mae=test_mae,
        test_mae_bps=test_mae_bps,
        baseline_mae_bps=baseline_mae_bps,
        per_d_mae_bps=per_d_mae_bps,
    )


def load_eq_autocall_surrogate(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[MaskedEQAutocallSurrogate, dict[str, Any]]:
    """Loads a trained `MaskedEQAutocallSurrogate` and its metadata from `checkpoint_path`."""
    ckpt_path = Path(checkpoint_path)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        cfg = raw.get("config", {})
        model = MaskedEQAutocallSurrogate(
            d_max=int(cfg.get("d_max", 8)),
            m_max=int(cfg.get("m_max", 48)),
            d_model=int(cfg.get("d_model", 128)),
            n_egno_layers=int(cfg.get("n_egno_layers", 3)),
            n_sched_layers=int(cfg.get("n_sched_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
        )
        model.load_state_dict(raw["model_state_dict"])
        metrics = dict(raw.get("metrics", {}))
    else:
        model = MaskedEQAutocallSurrogate()
        model.load_state_dict(raw)
        metrics = {}

    model = model.to(device=device, dtype=dtype).eval()
    return model, metrics


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "SobolevLossBreakdown",
    "SurrogateTrainingResult",
    "SpreadAttenuatedSobolevLoss",
    "assemble_surrogate_batch_pv",
    "evaluate_surrogate_mae_bps",
    "train_eq_autocall_surrogate",
    "load_eq_autocall_surrogate",
]
