"""
Deep Ensemble Training & OOD Calibration Pipeline (Phase D).

Trains K=5 independent CorrectionMLP models with different random seeds
for weight initialization and data shuffling, applies In-VRAM acceleration,
Adaptive Huber loss, 19D Sobolev regularization, and multivariable monotonicity
enforcement, and calibrates epistemic uncertainty thresholds for SR 26-2 OOD routing.

References:
    - Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable
      Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from deepvol.surrogates.correction_mlp import CorrectionMLP
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
from deepvol.training.train_correction import (
    CorrectionInputNormalizer,
    CorrectionOutputNormalizer,
    InVRAMDataLoader,
    compute_total_barrier_derivative,
)

DEFAULT_ENSEMBLE_CONFIG: Dict[str, Any] = {
    "K": 5,
    "seed_base": 42,
    "train_path": "data/autocall/train_120k_sobol_pde.npz",
    "val_path": "data/autocall/val_10k_sobol_pde.npz",
    "n_epochs": 70,
    "batch_size": 512,
    "hidden": 256,
    "n_layers": 4,
    "dropout": 0.0,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "patience": 30,
    "warmup_epochs": 3,
    "huber_beta_bps": 2.5,
    "device_str": "cuda",
    "in_dim": 19,
    "lambda_smooth": 0.05,
    "lambda_mono": 1.0,
    "gamma_frobenius": 1.80,
    "barrier_weight_alpha": 10.0,
    "barrier_weight_beta": 5.0,
    "weights_dir": "artifacts/weights",
    "weights_prefix": "autocall_correction_mlp_member_",
    "norm_in_path": "artifacts/scalers/correction_input_normalizer.npz",
    "norm_out_path": "artifacts/scalers/correction_output_normalizer.npz",
    "calibration_path": "artifacts/weights/ensemble_calibration.json",
    "resume": True,
}


def train_single_member(
    member_idx: int,
    seed: int,
    config: Dict[str, Any],
    train_loader: InVRAMDataLoader,
    X_val_t: torch.Tensor,
    pde_val_t: torch.Tensor,
    mc_val_t: torch.Tensor,
    norm_in: CorrectionInputNormalizer,
    norm_out: CorrectionOutputNormalizer,
    device: torch.device,
) -> str:
    """Trains a single CorrectionMLP ensemble member."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n--- Training Ensemble Member {member_idx + 1}/{config['K']} (seed={seed}) ---")

    in_dim = config.get("in_dim", 19)
    model = CorrectionMLP(
        in_dim=in_dim,
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    member_save_path = os.path.join(
        config["weights_dir"], f"{config['weights_prefix']}{member_idx}.pth"
    )

    best_val_score = float("inf")
    best_val_rmse = float("inf")
    best_state_dict = None
    mono_val_pct = 0.5

    # Checkpoint resumption
    if config.get("resume", False) and os.path.exists(member_save_path):
        saved = torch.load(member_save_path, map_location=device, weights_only=True)
        model.load_state_dict(saved)
        ema_model.module.load_state_dict(saved)
        best_state_dict = {k: v.cpu().clone() for k, v in saved.items()}
        # Initial evaluation
        eval_model = ema_model.module
        eval_model.eval()
        with torch.no_grad():
            preds_val = eval_model(X_val_t)
            delta_v_pred = norm_out.inverse_transform_tensor(preds_val)
            total_pred = pde_val_t + delta_v_pred.squeeze(-1).to(torch.float64)
            errors_bps = (total_pred - mc_val_t) * 10000.0
            best_val_rmse = float(torch.sqrt(torch.mean(errors_bps ** 2)).item())
            trimmed_mask = torch.abs(errors_bps) <= float(torch.quantile(torch.abs(errors_bps), 0.99).item())
            trimmed_rmse_bps = float(torch.sqrt(torch.mean(errors_bps[trimmed_mask] ** 2)).item())
        with torch.enable_grad():
            X_val_eval = X_val_t.clone().detach().requires_grad_(True)
            eval_preds = eval_model._forward_uncompiled(X_val_eval)
            eval_jac = torch.autograd.grad(eval_preds.sum(), X_val_eval, create_graph=False)[0]
            eval_df_dB = compute_total_barrier_derivative(eval_jac, X_val_eval, norm_in, norm_out)
            mono_val_pct = float((eval_df_dB > 1e-4).float().mean().item()) * 100.0
        best_val_score = best_val_rmse + 2.0 * max(0.0, (mono_val_pct - 0.2) / 100.0)
        print(
            f"Resumed Member {member_idx} weights from {member_save_path} | "
            f"Initial Val Raw: {best_val_rmse:.2f} bps | Trim: {trimmed_rmse_bps:.2f} bps | "
            f"Mono: {mono_val_pct:.2f}% (Score: {best_val_score:.4f})"
        )

    # Gentle warmup + cosine annealing schedule
    warmup_epochs = config.get("warmup_epochs", 3)
    n_epochs = config["n_epochs"]
    peak_lr = config["lr"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=config["weight_decay"]
    )

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    beta_bps = config.get("huber_beta_bps", 2.5)
    beta_normalized = (beta_bps / 10000.0) / float(norm_out.std[0])
    criterion = nn.SmoothL1Loss(beta=beta_normalized, reduction="none")

    patience = config["patience"]
    patience_counter = 0

    lambda_smooth = config.get("lambda_smooth", 0.05)
    lambda_mono = config.get("lambda_mono", 1.0)
    gamma_frobenius = config.get("gamma_frobenius", 1.80)

    t0 = time.perf_counter()

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        train_loss_accum = torch.tensor(0.0, device=device)
        train_count = 0

        for batch_x, batch_y, batch_w in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if lambda_smooth > 0 or lambda_mono > 0:
                batch_x.requires_grad_(True)
                preds = model._forward_uncompiled(batch_x)
                raw_loss = criterion(preds, batch_y)
                value_loss = (raw_loss * batch_w).mean()

                jac = torch.autograd.grad(preds.sum(), batch_x, create_graph=True)[0]

                reg_loss = torch.tensor(0.0, device=device)

                if lambda_smooth > 0:
                    jac_norm = torch.linalg.vector_norm(jac, dim=-1)
                    frobenius_penalty = torch.relu(jac_norm - gamma_frobenius).pow(2).mean()
                    reg_loss = reg_loss + lambda_smooth * frobenius_penalty

                if lambda_mono > 0:
                    df_dB = compute_total_barrier_derivative(jac, batch_x, norm_in, norm_out)
                    std_out_t = norm_out.get_std_tensor(device, batch_x.dtype)[0]
                    df_dB_norm = df_dB / std_out_t
                    mono_penalty = torch.relu(df_dB_norm).pow(2).mean()
                    reg_loss = reg_loss + lambda_mono * mono_penalty

                loss = value_loss + reg_loss
            else:
                preds = model(batch_x)
                raw_loss = criterion(preds, batch_y)
                loss = (raw_loss * batch_w).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ema_model.update_parameters(model)
            train_loss_accum += loss.detach() * len(batch_x)
            train_count += len(batch_x)

        train_loss = float(train_loss_accum.item()) / max(1, train_count)
        scheduler.step()

        # Vectorized GPU Validation using EMA Model
        eval_model = ema_model.module
        eval_model.eval()
        with torch.no_grad():
            preds_val = eval_model(X_val_t)
            delta_v_pred = norm_out.inverse_transform_tensor(preds_val)
            total_pred = pde_val_t + delta_v_pred.squeeze(-1).to(torch.float64)
            errors_bps = (total_pred - mc_val_t) * 10000.0

            val_rmse_bps = float(torch.sqrt(torch.mean(errors_bps ** 2)).item())
            val_mae_bps = float(torch.mean(torch.abs(errors_bps)).item())

            p99_thresh = float(torch.quantile(torch.abs(errors_bps), 0.99).item())
            trimmed_mask = torch.abs(errors_bps) <= p99_thresh
            trimmed_rmse_bps = float(torch.sqrt(torch.mean(errors_bps[trimmed_mask] ** 2)).item())

        # Vectorized GPU Monotonicity Validation (evaluate every 5 epochs or on candidate bests)
        if epoch % 5 == 0 or epoch == 1 or val_rmse_bps < best_val_rmse:
            with torch.enable_grad():
                X_val_eval = X_val_t.clone().detach().requires_grad_(True)
                eval_preds = eval_model._forward_uncompiled(X_val_eval)
                eval_jac = torch.autograd.grad(eval_preds.sum(), X_val_eval, create_graph=False)[0]
                eval_df_dB = compute_total_barrier_derivative(eval_jac, X_val_eval, norm_in, norm_out)
                mono_val_pct = float((eval_df_dB > 1e-4).float().mean().item()) * 100.0

        if epoch % 10 == 0 or epoch == 1 or trimmed_rmse_bps < 1.0:
            print(
                f"Member {member_idx} | Ep {epoch:2d}/{config['n_epochs']} | "
                f"Loss: {train_loss:.5f} | Raw: {val_rmse_bps:.2f} bps | "
                f"Trim: {trimmed_rmse_bps:.2f} bps | MAE: {val_mae_bps:.2f} bps | "
                f"Mono: {mono_val_pct:.2f}%"
            )

        val_score = val_rmse_bps + 2.0 * max(0.0, (mono_val_pct - 0.2) / 100.0)

        if val_score < best_val_score:
            best_val_score = val_score
            best_val_rmse = val_rmse_bps
            best_state_dict = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
            patience_counter = 0
            os.makedirs(os.path.dirname(member_save_path), exist_ok=True)
            torch.save(best_state_dict, member_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop member {member_idx} at epoch {epoch} (best score: {best_val_score:.4f}, best RMSE: {best_val_rmse:.2f} bps)")
                break

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    elapsed = time.perf_counter() - t0
    print(f"Member {member_idx} finished in {elapsed:.1f}s | Best Val RMSE: {best_val_rmse:.2f} bps")
    return member_save_path


def train_ensemble(config: Dict[str, Any]) -> Tuple[CorrectionEnsemble, Dict[str, float]]:
    """
    Trains all K ensemble members and calibrates OOD uncertainty thresholds.
    """
    from deepvol.utils.gpu_lock import acquire_gpu_lock
    acquire_gpu_lock()

    device = torch.device(
        config.get("device_str", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"Training CorrectionEnsemble (K={config['K']}) on {device}...")

    # Load and fit normalizers
    train_data = np.load(config["train_path"])
    X_train = CorrectionInputNormalizer.build_feature_matrix(train_data)
    Y_train = (train_data["npv"] - train_data["pde_npv"]).reshape(-1, 1)

    norm_in = CorrectionInputNormalizer().fit(X_train)
    norm_out = CorrectionOutputNormalizer().fit(Y_train)

    os.makedirs(os.path.dirname(config["norm_in_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["norm_out_path"]), exist_ok=True)
    norm_in.save(config["norm_in_path"])
    norm_out.save(config["norm_out_path"])

    # Compute barrier proximity weights
    alpha = config.get("barrier_weight_alpha", 10.0)
    beta = config.get("barrier_weight_beta", 5.0)
    B_train = np.maximum(np.asarray(train_data["B"], dtype=np.float32), 1e-8)
    weights_train = 1.0 + alpha * np.exp(-beta * np.abs(np.log(B_train)))

    # In-VRAM resident training buffers
    X_train_norm = norm_in.transform(X_train)
    Y_train_norm = norm_out.transform(Y_train)
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train_norm, dtype=torch.float32, device=device)
    W_train_t = torch.tensor(weights_train, dtype=torch.float32, device=device).reshape(-1, 1)

    train_loader = InVRAMDataLoader(
        X_train_t, Y_train_t, W_train_t, batch_size=config["batch_size"]
    )

    # In-VRAM validation buffers
    val_data = np.load(config["val_path"])
    X_val = CorrectionInputNormalizer.build_feature_matrix(val_data)
    X_val_norm = norm_in.transform(X_val)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    pde_val_t = torch.tensor(val_data["pde_npv"], dtype=torch.float64, device=device)
    mc_val_t = torch.tensor(val_data["npv"], dtype=torch.float64, device=device)

    K = config["K"]
    seed_base = config.get("seed_base", 42)
    member_paths: List[str] = []

    for k in range(K):
        path = train_single_member(
            member_idx=k,
            seed=seed_base + k,
            config=config,
            train_loader=train_loader,
            X_val_t=X_val_t,
            pde_val_t=pde_val_t,
            mc_val_t=mc_val_t,
            norm_in=norm_in,
            norm_out=norm_out,
            device=device,
        )
        member_paths.append(path)

    # Construct and load full ensemble
    print("\n--- Assembling & Calibrating Deep Ensemble ---")
    ensemble = CorrectionEnsemble(
        K=K,
        in_dim=config["in_dim"],
        hidden=config["hidden"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)
    ensemble.load_members(member_paths, device=device)
    ensemble.eval()

    # Fast Vectorized Ensemble Validation
    with torch.no_grad():
        mean_pred_norm, std_pred_norm = ensemble(X_val_t)
        # Denormalize predictions (strict double precision)
        delta_v_mean = norm_out.inverse_transform_tensor(mean_pred_norm)
        total_pred_ens = pde_val_t + delta_v_mean.squeeze(-1).to(torch.float64)
        errors_bps = (total_pred_ens - mc_val_t) * 10000.0

        std_bps = (
            std_pred_norm * norm_out.get_std_tensor(device, torch.float32)[0]
        ).abs() * 10000.0

        raw_rmse = float(torch.sqrt(torch.mean(errors_bps ** 2)).item())
        raw_mae = float(torch.mean(torch.abs(errors_bps)).item())

        p95_error = float(torch.quantile(torch.abs(errors_bps), 0.95).item())
        p99_error = float(torch.quantile(torch.abs(errors_bps), 0.99).item())
        p100_error = float(torch.max(torch.abs(errors_bps)).item())

        trimmed_mask = torch.abs(errors_bps) <= p99_error
        trimmed_rmse = float(torch.sqrt(torch.mean(errors_bps[trimmed_mask] ** 2)).item())

        # ITM Barrier (B > 1.05) metrics
        B_val = torch.tensor(val_data["B"], device=device)
        itm_mask = B_val > 1.05
        itm_rmse = float(torch.sqrt(torch.mean(errors_bps[itm_mask] ** 2)).item()) if itm_mask.any() else 0.0

        # High-sigma (sigma > 0.7) metrics
        sigma_val = torch.tensor(val_data["sigma"], device=device)
        high_sig_mask = sigma_val > 0.7
        high_sig_rmse = float(torch.sqrt(torch.mean(errors_bps[high_sig_mask] ** 2)).item()) if high_sig_mask.any() else 0.0

        # Calibrate OOD uncertainty threshold tau_ood
        # tau_ood set at P99 of validation uncertainty distribution
        tau_ood_p99 = float(torch.quantile(std_bps, 0.99).item())
        tau_ood_p95 = float(torch.quantile(std_bps, 0.95).item())

        # Check recall on worst 1% error outliers
        outlier_mask = torch.abs(errors_bps) > p99_error
        flagged_outliers = (std_bps.squeeze(-1) > tau_ood_p95) & outlier_mask
        outlier_recall = float(flagged_outliers.sum().item()) / max(1, int(outlier_mask.sum().item()))

    calibration_results: Dict[str, float] = {
        "raw_rmse_bps": raw_rmse,
        "trimmed_rmse_bps": trimmed_rmse,
        "raw_mae_bps": raw_mae,
        "p95_error_bps": p95_error,
        "p99_error_bps": p99_error,
        "p100_error_bps": p100_error,
        "itm_barrier_rmse_bps": itm_rmse,
        "high_sigma_rmse_bps": high_sig_rmse,
        "tau_ood_p99_bps": tau_ood_p99,
        "tau_ood_p95_bps": tau_ood_p95,
        "outlier_recall_p95": outlier_recall,
    }

    print("\n=======================================================")
    print("ENSEMBLE VALIDATION & OOD CALIBRATION SUMMARY:")
    print(f"  Ensemble Raw RMSE:        {raw_rmse:.2f} bps")
    print(f"  Ensemble Trimmed RMSE:    {trimmed_rmse:.2f} bps")
    print(f"  Ensemble MAE:             {raw_mae:.2f} bps")
    print(f"  P95 Absolute Error:       {p95_error:.2f} bps")
    print(f"  P99 Absolute Error:       {p99_error:.2f} bps")
    print(f"  P100 (Worst Outlier):     {p100_error:.2f} bps")
    print(f"  ITM Barrier (B>1.05) RMSE:{itm_rmse:.2f} bps")
    print(f"  High-sigma (sig>0.7) RMSE:{high_sig_rmse:.2f} bps")
    print(f"  Calibrated tau_ood (P99): {tau_ood_p99:.2f} bps")
    print(f"  Calibrated tau_ood (P95): {tau_ood_p95:.2f} bps")
    print(f"  Outlier Recall @ P95:     {outlier_recall:.1%}")
    print("=======================================================\n")

    os.makedirs(os.path.dirname(config["calibration_path"]), exist_ok=True)
    with open(config["calibration_path"], "w") as f:
        json.dump(calibration_results, f, indent=2)
    print(f"Saved ensemble calibration to {config['calibration_path']}")

    # Also save member 0 as default single model checkpoint for backward compatibility
    import shutil
    shutil.copy(member_paths[0], "artifacts/weights/autocall_correction_mlp.pth")

    return ensemble, calibration_results


if __name__ == "__main__":
    train_ensemble(DEFAULT_ENSEMBLE_CONFIG)
