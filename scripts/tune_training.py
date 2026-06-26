# ruff: noqa: E402
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import IVSurfaceNormalizer
from deepvol.calibration.active_learning import (
    ParameterNormalizerGrey,
    compute_bs_price_delta_vega,
    make_spatial_input,
)


def run_experiment(w_iv=1.0, w_price=1.0, w_delta=1.0, w_vega=1.0, lr=1e-3, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join(repo_root, "data", "grey_al_dataset.npz")
    data = np.load(dataset_path)
    final_params = torch.tensor(data["params"], dtype=torch.float32)
    final_ivs = torch.tensor(data["ivs"], dtype=torch.float32)

    # Grid parameters
    from deepvol.calibration.grey_calibrator import GreyRoughBergomiCalibrator

    calib = GreyRoughBergomiCalibrator(steps=50, paths=1000).to(device)
    T_grid = calib.T_grid
    K_grid = calib.K_grid

    num_T = len(T_grid)
    num_K = len(K_grid)
    modes1 = num_T
    modes2 = num_K // 2 + 1

    # Fit normalizers
    param_norm = ParameterNormalizerGrey().fit(final_params.numpy())
    iv_norm = IVSurfaceNormalizer().fit(final_ivs.numpy())

    final_params_norm = torch.tensor(
        param_norm.transform(final_params.numpy()), dtype=torch.float32
    )
    final_ivs_norm = torch.tensor(
        iv_norm.transform(final_ivs.numpy()), dtype=torch.float32
    )

    # Split
    N_total = len(final_params)
    torch.manual_seed(42)
    perm = torch.randperm(N_total)
    train_split = int(0.8 * N_total)

    train_idx = perm[:train_split]
    val_idx = perm[train_split:]

    tr_ds = TensorDataset(final_params_norm[train_idx], final_ivs_norm[train_idx])
    val_ds = TensorDataset(final_params_norm[val_idx], final_ivs_norm[val_idx])

    train_dl = DataLoader(tr_ds, batch_size=256, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=True)

    spatial = make_spatial_input(T_grid, K_grid, device)

    model = MirrorPaddedFNO2d(modes1=modes1, modes2=modes2, param_dim=5, width=64).to(
        device
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    K_grid_t = torch.tensor(K_grid.cpu().numpy(), dtype=torch.float64, device=device)
    T_grid_t = torch.tensor(T_grid.cpu().numpy(), dtype=torch.float64, device=device)

    best_val_mse = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        for X_b, Y_b in train_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)

            sp = spatial.expand(B, -1, -1, -1)
            pred_norm = model(sp, X_b)

            loss_iv = F.huber_loss(pred_norm, Y_b, delta=1.0)

            loss = w_iv * loss_iv

            if w_price > 0 or w_delta > 0 or w_vega > 0:
                pred_iv = iv_norm.inverse_transform_tensor(pred_norm)
                target_iv = iv_norm.inverse_transform_tensor(Y_b)

                P_pred, delta_pred, vega_pred = compute_bs_price_delta_vega(
                    pred_iv, K_grid_t, T_grid_t
                )
                P_target, delta_target, vega_target = compute_bs_price_delta_vega(
                    target_iv, K_grid_t, T_grid_t
                )

                if w_price > 0:
                    loss += w_price * F.huber_loss(P_pred, P_target, delta=1.0)
                if w_delta > 0:
                    loss += w_delta * F.huber_loss(delta_pred, delta_target, delta=1.0)
                if w_vega > 0:
                    loss += w_vega * F.huber_loss(vega_pred, vega_target, delta=1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for X_b, Y_b in val_dl:
                X_b = X_b.to(device)
                Y_b = Y_b.to(device)
                B = X_b.size(0)

                sp = spatial.expand(B, -1, -1, -1)
                pred_norm = model(sp, X_b)

                pred_iv = iv_norm.inverse_transform_tensor(pred_norm)
                target_iv = iv_norm.inverse_transform_tensor(Y_b)

                val_mse += F.mse_loss(pred_iv, target_iv).item() * B

        val_mse /= len(val_dl.dataset)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_pred = pred_iv
            best_target = target_iv
            best_val_norm_mse = F.mse_loss(pred_norm, Y_b).item()

    print(
        f"w_iv={w_iv}, w_price={w_price}, w_delta={w_delta}, w_vega={w_vega}, lr={lr}, epochs={epochs}"
    )
    print(f"  Best Val MSE (denorm): {best_val_mse:.4e}")
    print(f"  Best Val MSE (norm): {best_val_norm_mse:.4e}")

    print(
        f"  Pred mean: {best_pred.mean().item():.4f}, std: {best_pred.std().item():.4f}"
    )
    print(
        f"  Target mean: {best_target.mean().item():.4f}, std: {best_target.std().item():.4f}"
    )
    return best_val_mse


if __name__ == "__main__":
    # Experiment 1: Sobolev with 150 epochs, lr=1e-3, width=64
    run_experiment(w_iv=1.0, w_price=1.0, w_delta=1.0, w_vega=1.0, lr=1e-3, epochs=150)
