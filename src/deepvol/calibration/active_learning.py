# ruff: noqa: E402
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from deepvol.utils.gpu_lock import acquire_gpu_lock
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration.grey_calibrator import GreyRoughBergomiCalibrator
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer


class ParameterNormalizerGrey(ParameterNormalizer):
    """
    Z-score normalizer for 5-dimensional Grey Rough Bergomi parameter vector [v0, H, eta, rho, beta].
    """

    PARAM_NAMES = ["v0", "H", "eta", "rho", "beta"]


def generate_pool(size=50000, seed=42):
    """
    Generates a pool of random parameters within the specified ranges:
      - v0: [0.01, 0.16]
      - H: [0.05, 0.4]
      - eta: [0.5, 2.0]
      - rho: [-0.95, -0.2]
      - beta: [0.5, 1.0]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    v0 = torch.rand(size) * (0.16 - 0.01) + 0.01
    H = torch.rand(size) * (0.4 - 0.05) + 0.05
    eta = torch.rand(size) * (2.0 - 0.5) + 0.5
    rho = torch.rand(size) * (-0.2 - (-0.95)) - 0.95
    beta = torch.rand(size) * (1.0 - 0.5) + 0.5

    pool = torch.stack([v0, H, eta, rho, beta], dim=1)
    return pool


def query_calibrator(calibrator, params, batch_size=10):
    """
    Queries the Monte Carlo calibrator to generate true IV surfaces.
    Uses batching and cache clearing to prevent VRAM OOM.
    """
    num_samples = params.shape[0]
    iv_surfaces = []

    for idx in range(0, num_samples, batch_size):
        batch_params = params[idx : idx + batch_size]
        batch_params_cuda = batch_params.to(device="cuda", dtype=torch.float64)
        with torch.no_grad():
            batch_iv = calibrator(batch_params_cuda)
        iv_surfaces.append(batch_iv.cpu())
        torch.cuda.empty_cache()

    return torch.cat(iv_surfaces, dim=0)


def make_spatial_input(T_grid, K_grid, device):
    """
    Prepares the normalized coordinate grid mesh for the FNO model.
    """
    T_norm = (T_grid - T_grid.mean()) / T_grid.std()
    K_norm = (K_grid - K_grid.mean()) / K_grid.std()
    T_mesh, K_mesh = np.meshgrid(
        T_norm.cpu().numpy(), K_norm.cpu().numpy(), indexing="ij"
    )
    spatial = np.stack([T_mesh, K_mesh], axis=-1)[None]  # (1, nT, nK, 2)
    return torch.tensor(spatial, dtype=torch.float32, device=device)


def get_bootstrap_loader(X_data, Y_data, batch_size=256):
    """
    Draws a bootstrap sample from the dataset and returns a DataLoader.
    """
    num_samples = len(X_data)
    indices = torch.randint(0, num_samples, (num_samples,))
    X_boot = X_data[indices]
    Y_boot = Y_data[indices]
    ds = TensorDataset(X_boot, Y_boot)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)


def train_ensemble_model(model, tr_dl, spatial, epochs=50, device="cuda"):
    """
    Helper function to train a single ensemble FNO member.
    """
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(epochs):
        for X_b, Y_b in tr_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)

            sp = spatial.expand(B, -1, -1, -1)
            pred = model(sp, X_b)
            loss = F.huber_loss(pred, Y_b, delta=1.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()


def predict_with_ensemble(
    models, params, param_norm, iv_norm, spatial, batch_size=2048, device="cuda"
):
    """
    Predict IVs on parameters and compute the ensemble variance across the models.
    """
    for m in models:
        m.eval()
        m.to(device)

    num_samples = params.shape[0]
    all_preds = [[] for _ in range(len(models))]

    with torch.no_grad():
        for idx in range(0, num_samples, batch_size):
            batch_params = params[idx : idx + batch_size].numpy()
            batch_norm_params = param_norm.transform(batch_params)
            X_b = torch.tensor(batch_norm_params, dtype=torch.float32, device=device)
            B = X_b.size(0)
            sp = spatial.expand(B, -1, -1, -1)

            for m_idx, model in enumerate(models):
                pred_norm = model(sp, X_b)
                pred_iv = iv_norm.inverse_transform_tensor(pred_norm)
                all_preds[m_idx].append(pred_iv.cpu())

    concatenated_preds = [torch.cat(p, dim=0) for p in all_preds]
    stacked = torch.stack(
        concatenated_preds, dim=0
    )  # (num_models, num_samples, num_T, num_K)

    var_surfaces = torch.var(stacked, dim=0)  # (num_samples, num_T, num_K)
    variance_scores = var_surfaces.mean(dim=(1, 2))  # (num_samples,)

    return stacked, variance_scores


def compute_bs_price_delta_vega(sigma, K_grid, T_grid):
    """
    Computes analytical Black-Scholes price, Delta, and Vega.
    Operates strictly in torch.float64 internally to prevent gradient noise and precision loss.
    """
    sigma_f64 = sigma.to(torch.float64)
    K_grid_f64 = K_grid.to(torch.float64)
    T_grid_f64 = T_grid.to(torch.float64)

    B = sigma_f64.shape[0]
    num_T = len(T_grid_f64)
    num_K = len(K_grid_f64)

    T_mesh = (
        T_grid_f64[:, None]
        .expand(-1, num_K)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .to(sigma_f64.device)
    )
    k_mesh = (
        K_grid_f64[None, :]
        .expand(num_T, -1)
        .unsqueeze(0)
        .expand(B, -1, -1)
        .to(sigma_f64.device)
    )
    K_mesh = torch.exp(k_mesh)

    is_call = k_mesh >= 0.0

    sigma_safe = torch.clamp(sigma_f64, min=0.01)
    T_safe = torch.clamp(T_mesh, min=1e-8)

    sqrt_T = torch.sqrt(T_safe)
    d1 = (-k_mesh + 0.5 * (sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    SQRT_2 = 1.4142135623730951
    SQRT_2PI = 2.5066282746310005

    phi_d1 = 0.5 * (1.0 + torch.erf(d1 / SQRT_2))
    phi_d2 = 0.5 * (1.0 + torch.erf(d2 / SQRT_2))

    call_price = phi_d1 - K_mesh * phi_d2
    put_price = K_mesh * (1.0 - phi_d2) - (1.0 - phi_d1)

    price = torch.where(is_call, call_price, put_price)

    call_delta = phi_d1
    put_delta = phi_d1 - 1.0
    delta = torch.where(is_call, call_delta, put_delta)

    n_prime_d1 = torch.exp(-0.5 * (d1**2)) / SQRT_2PI
    vega = sqrt_T * n_prime_d1

    return price.to(sigma.dtype), delta.to(sigma.dtype), vega.to(sigma.dtype)


def train_sobolev_fno(
    model,
    train_dl,
    val_dl,
    spatial,
    K_grid,
    T_grid,
    iv_norm,
    epochs=100,
    device="cuda",
    w_iv=1.0,
    w_price=1.0,
    w_delta=1.0,
    w_vega=1.0,
    weights_best_path=None,
    weights_prod_path=None,
):
    """
    Trains the FNO model using Sobolev training.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse = float("inf")

    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val MSE':>10}")
    print("  " + "-" * 35)

    K_grid_t = torch.tensor(K_grid, dtype=torch.float64, device=device)
    T_grid_t = torch.tensor(T_grid, dtype=torch.float64, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for X_b, Y_b in train_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)

            sp = spatial.expand(B, -1, -1, -1)
            pred_norm = model(sp, X_b)

            # 1. Huber loss on normalized IV
            loss_iv = F.huber_loss(pred_norm, Y_b, delta=1.0)

            # 2. Sobolev losses (computed on denormalized values)
            pred_iv = iv_norm.inverse_transform_tensor(pred_norm)
            target_iv = iv_norm.inverse_transform_tensor(Y_b)

            P_pred, delta_pred, vega_pred = compute_bs_price_delta_vega(
                pred_iv, K_grid_t, T_grid_t
            )
            P_target, delta_target, vega_target = compute_bs_price_delta_vega(
                target_iv, K_grid_t, T_grid_t
            )

            loss_price = F.huber_loss(P_pred, P_target, delta=1.0)
            loss_delta = F.huber_loss(delta_pred, delta_target, delta=1.0)
            loss_vega = F.huber_loss(vega_pred, vega_target, delta=1.0)

            # Combine losses
            loss = (
                w_iv * loss_iv
                + w_price * loss_price
                + w_delta * loss_delta
                + w_vega * loss_vega
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * B

        tr_loss /= len(train_dl.dataset)
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
            if weights_best_path:
                os.makedirs(os.path.dirname(weights_best_path), exist_ok=True)
                torch.save(model.state_dict(), weights_best_path)

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"  {ep:>5} | {tr_loss:>10.6f} | {val_mse:>10.6e}")

    # Save final model weights
    if weights_prod_path:
        os.makedirs(os.path.dirname(weights_prod_path), exist_ok=True)
        torch.save(model.state_dict(), weights_prod_path)

    print(f"\n  Final Best Validation MSE: {best_val_mse:.4e}")
    return best_val_mse


def run_active_learning(smoke=False):
    """
    Main active learning and Sobolev training pipeline.
    """
    print("--- Acquiring GPU Lock ---")
    acquire_gpu_lock()
    print("--- GPU Lock Acquired ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Grid sizes and hyperparams mapping
    pool_size = 50000
    seed_size = 2000
    query_size = 3000
    ensemble_epochs = 50
    final_epochs = 100
    mc_steps = 200
    mc_paths = 15000

    if smoke:
        print("Running in SMOKE mode for validation...")
        pool_size = 500
        seed_size = 50
        query_size = 30
        ensemble_epochs = 5
        final_epochs = 10
        mc_steps = 50
        mc_paths = 1000

    # Paths setup
    artifacts_dir = os.path.join(repo_root, "artifacts")
    weights_best_path = os.path.join(artifacts_dir, "weights", "fno_grey_best.pth")
    weights_prod_path = os.path.join(
        artifacts_dir, "weights", "fno_grey_final_prod.pth"
    )
    param_norm_path = os.path.join(artifacts_dir, "models", "param_normalizer_grey.npz")
    iv_norm_path = os.path.join(artifacts_dir, "models", "iv_normalizer_grey.npz")

    # 1. Initialize Monte Carlo Calibrator
    calibrator = GreyRoughBergomiCalibrator(steps=mc_steps, paths=mc_paths).to(device)
    T_grid = calibrator.T_grid
    K_grid = calibrator.K_grid

    # Compute modes dynamically from grids to avoid einsum shape mismatches
    num_T = len(T_grid)
    num_K = len(K_grid)
    modes1 = num_T
    modes2 = num_K // 2 + 1

    # 2. Generate Pool
    print(f"Generating pool of {pool_size} parameter combinations...")
    pool = generate_pool(size=pool_size)

    # 3. Select Seed
    shuffled_indices = torch.randperm(pool_size)
    seed_indices = shuffled_indices[:seed_size]
    remaining_indices = shuffled_indices[seed_size:]

    seed_params = pool[seed_indices]
    remaining_params = pool[remaining_indices]

    # 4. Query MC Simulator for Seed
    print(f"Querying Monte Carlo simulator for seed dataset ({seed_size} samples)...")
    t0 = time.time()
    seed_ivs = query_calibrator(calibrator, seed_params, batch_size=10)
    print(f"Seed generation completed in {time.time() - t0:.2f}s")

    # 5. Normalize Seed
    param_norm = ParameterNormalizerGrey().fit(seed_params.numpy())
    iv_norm = IVSurfaceNormalizer().fit(seed_ivs.numpy())

    # Transform Seed
    seed_params_norm = torch.tensor(
        param_norm.transform(seed_params.numpy()), dtype=torch.float32
    )
    seed_ivs_norm = torch.tensor(
        iv_norm.transform(seed_ivs.numpy()), dtype=torch.float32
    )

    # 6. Train Ensemble of 3 Models
    print("Training FNO ensemble of 3 models...")
    spatial = make_spatial_input(T_grid, K_grid, device)

    ensemble_models = []
    for i in range(3):
        print(f"  Training ensemble model {i + 1}/3...")
        model = MirrorPaddedFNO2d(modes1=modes1, modes2=modes2, param_dim=5)
        # Use bootstrap sampling and random seed initialization for diversity
        torch.manual_seed(42 + i)
        tr_dl = get_bootstrap_loader(seed_params_norm, seed_ivs_norm, batch_size=256)
        train_ensemble_model(
            model, tr_dl, spatial, epochs=ensemble_epochs, device=device
        )
        ensemble_models.append(model)

    # 7. Predict on Pool & Compute Variance
    print(f"Predicting on remaining {len(remaining_params)} parameters...")
    stacked_preds, variance_scores = predict_with_ensemble(
        ensemble_models, remaining_params, param_norm, iv_norm, spatial, device=device
    )

    # 8. Query Top 3,000 parameter combinations with highest variance
    top_var_indices = torch.topk(variance_scores, k=query_size).indices
    queried_params = remaining_params[top_var_indices]

    print(f"Querying Monte Carlo simulator for top {query_size} uncertainty samples...")
    t0 = time.time()
    queried_ivs = query_calibrator(calibrator, queried_params, batch_size=10)
    print(f"Query generation completed in {time.time() - t0:.2f}s")

    # 9. Combine datasets
    final_params = torch.cat([seed_params, queried_params], dim=0)
    final_ivs = torch.cat([seed_ivs, queried_ivs], dim=0)
    print(f"Combined final dataset size: {final_params.shape[0]} samples")

    # Save the combined dataset to disk
    dataset_dir = os.path.join(repo_root, "data")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, "grey_al_dataset.npz")
    np.savez(dataset_path, params=final_params.numpy(), ivs=final_ivs.numpy())
    print(f"Combined dataset saved to {dataset_path}")

    # 10. Refit and save normalizers on combined dataset
    param_norm = ParameterNormalizerGrey().fit(final_params.numpy())
    iv_norm = IVSurfaceNormalizer().fit(final_ivs.numpy())

    os.makedirs(os.path.dirname(param_norm_path), exist_ok=True)
    param_norm.save(param_norm_path)
    iv_norm.save(iv_norm_path)
    print(f"Normalizers saved to {param_norm_path} and {iv_norm_path}")

    # Normalize final dataset
    final_params_norm = torch.tensor(
        param_norm.transform(final_params.numpy()), dtype=torch.float32
    )
    final_ivs_norm = torch.tensor(
        iv_norm.transform(final_ivs.numpy()), dtype=torch.float32
    )

    # 11. Train/Val Split (80/20)
    N_total = len(final_params)
    perm = torch.randperm(N_total)
    train_split = int(0.8 * N_total)

    train_idx = perm[:train_split]
    val_idx = perm[train_split:]

    tr_ds = TensorDataset(final_params_norm[train_idx], final_ivs_norm[train_idx])
    val_ds = TensorDataset(final_params_norm[val_idx], final_ivs_norm[val_idx])

    train_dl = DataLoader(tr_ds, batch_size=256, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=True)

    # 12. Train final model with Sobolev FNO Training
    print("Training final FNO model using Sobolev training...")
    final_model = MirrorPaddedFNO2d(modes1=modes1, modes2=modes2, param_dim=5)
    best_val_mse = train_sobolev_fno(
        model=final_model,
        train_dl=train_dl,
        val_dl=val_dl,
        spatial=spatial,
        K_grid=K_grid.cpu().numpy(),
        T_grid=T_grid.cpu().numpy(),
        iv_norm=iv_norm,
        epochs=final_epochs,
        device=device,
        w_iv=1.0,
        w_price=1.0,
        w_delta=1.0,
        w_vega=1.0,
        weights_best_path=weights_best_path,
        weights_prod_path=weights_prod_path,
    )

    print("Active learning workflow completed successfully!")
    print(f"Best Validation MSE: {best_val_mse:.4e}")
    return best_val_mse


if __name__ == "__main__":
    smoke_mode = "--smoke" in sys.argv or "--test" in sys.argv
    run_active_learning(smoke=smoke_mode)
