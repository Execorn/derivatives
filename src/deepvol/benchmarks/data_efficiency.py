#!/usr/bin/env python
"""
data_efficiency.py — Benchmark data efficiency across GPE, PCE, and FNO models.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Monkeypatch numpy.bool to fix chaospy/numpoly compatibility with newer numpy
np.bool = np.bool_
import chaospy as cp

from sklearn.linear_model import Ridge, RidgeCV, MultiTaskLassoCV


# Ensure the 'src' directory is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from deepvol.surrogates.fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization
from deepvol.surrogates.normalizers import ParameterNormalizerHeston, IVSurfaceNormalizer
import gpytorch


# ─── GPyTorch Batch SVGP Model ───────────────────────────────────────────────

class BatchSVGPModel(gpytorch.models.ApproximateGP):
    """
    Batched Stochastic Variational Gaussian Process (SVGP) model.
    Trains independent GP models (one for each grid point of the surface)
    in parallel using GPyTorch's batch mode.
    """
    def __init__(self, inducing_points):
        # inducing_points shape: (n_grid_points, num_inducing, n_params)
        batch_shape = torch.Size([inducing_points.size(0)])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )

    def forward(self, x):
        # x shape: (n_grid_points, B, n_params)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ─── Helper Functions ────────────────────────────────────────────────────────

def _make_spatial_input(T_grid, K_grid, device):
    """Generates the spatial coordinate input mesh normalized for the FNO."""
    T_norm = (T_grid - T_grid.mean()) / T_grid.std()
    K_norm = (K_grid - K_grid.mean()) / K_grid.std()
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing='ij')
    spatial = np.stack([T_mesh, K_mesh], axis=-1)[None]  # (1, nT, nK, 2)
    return torch.tensor(spatial, dtype=torch.float32, device=device)


def compute_metrics(pred, target):
    """Computes MSE, MAE, and R2 score over the implied volatility surface."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    var_target = np.var(target_flat)
    r2 = 1.0 - (mse / var_target) if var_target > 1e-8 else 0.0
    
    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Data Efficiency Benchmark for DeepVol")
    parser.add_argument("-N", "--n_samples", type=int, default=10000,
                        help="Number of subset samples to load (default: 10000)")
    parser.add_argument("--model", type=str, choices=["fno", "gpe", "pce", "all"], default="all",
                        help="Model to benchmark (default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run in CPU-only smoke test mode with small epochs and data")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for FNO training (default: 50)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for FNO and GPE (default: 1024)")
    parser.add_argument("--lr", type=float, default=8e-4,
                        help="Learning rate for FNO (default: 8e-4)")
    parser.add_argument("--gp_epochs", type=int, default=15,
                        help="Number of epochs for GPE training (default: 15)")
    parser.add_argument("--gp_lr", type=float, default=0.02,
                        help="Learning rate for GPE (default: 0.02)")
    parser.add_argument("--gp_inducing", type=int, default=100,
                        help="Number of inducing points for GPE (default: 100)")
    parser.add_argument("--pce_order", type=int, default=3,
                        help="Polynomial Chaos Expansion order (default: 3)")
    parser.add_argument("--pce_reg", type=str, choices=["lasso", "ridge", "ols", "auto"], default="auto",
                        help="Regularization for PCE regression (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dataset_path", type=str, default="data/HestonDataset_v1.npz",
                        help="Path to Heston dataset (default: data/HestonDataset_v1.npz)")
    return parser.parse_args()


# ─── Main Benchmarking Function ──────────────────────────────────────────────

def main():
    args = parse_args()
    
    # Handle smoke mode defaults
    if args.smoke:
        print("⚡ Running in SMOKE mode: forcing minimal epochs and data on CPU")
        args.n_samples = 100
        args.epochs = 2
        args.gp_epochs = 2
        args.gp_inducing = 20
        args.pce_order = 2
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Using device: {device}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ── Load Dataset ─────────────────────────────────────────────────────────
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Heston dataset not found at: {args.dataset_path}")
        
    print(f"Loading Heston dataset from {args.dataset_path}...")
    dataset = np.load(args.dataset_path)
    params_all = dataset["params"]  # (N_total, 5) -> [kappa, theta, sigma, rho, v0]
    iv_all = dataset["iv"]          # (N_total, 8, 11)
    T_grid = dataset["T_grid"]      # (8,)
    K_grid = dataset["K_grid"]      # (11,)
    
    total_available = params_all.shape[0]
    N = min(args.n_samples, total_available)
    if N < 13:
        N = min(13, total_available)
    print(f"Dataset loaded. Total samples: {total_available:,}. Benchmarking with N = {N:,}")
    
    # Create cache directory if it does not exist
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize rng for reproducibility in downstream code
    rng = np.random.default_rng(args.seed)
    
    subset_path = os.path.join(cache_dir, f"subset_N{N}_seed{args.seed}.npz")
    if os.path.exists(subset_path):
        print(f"Loading cached subset from {subset_path}")
        subset_data = np.load(subset_path)
        X_tr = subset_data["X_tr"]
        X_va = subset_data["X_va"]
        Y_tr = subset_data["Y_tr"]
        Y_va = subset_data["Y_va"]
    else:
        print(f"Generating and saving subset to {subset_path}")
        perm = rng.permutation(total_available)
        params = params_all[perm[:N]]
        iv = iv_all[perm[:N]]
        
        # Train/Validation Split (80% / 20%)
        split = int(0.8 * N)
        if split < 10:
            split = 10
        if split > N:
            split = N
        X_tr, X_va = params[:split], params[split:]
        Y_tr, Y_va = iv[:split], iv[split:]
        np.savez(subset_path, X_tr=X_tr, X_va=X_va, Y_tr=Y_tr, Y_va=Y_va)
    
    print(f"Split sizes: Train={X_tr.shape[0]:,}, Validation={X_va.shape[0]:,}")
    
    # ── Fit Normalizers ──────────────────────────────────────────────────────
    print("Fitting normalizers on train split...")
    param_normalizer = ParameterNormalizerHeston().fit(X_tr)
    iv_normalizer = IVSurfaceNormalizer().fit(Y_tr)
    
    X_tr_n = param_normalizer.transform(X_tr)
    X_va_n = param_normalizer.transform(X_va)
    Y_tr_n = iv_normalizer.transform(Y_tr)
    Y_va_n = iv_normalizer.transform(Y_va)
    
    n_params = X_tr.shape[1]
    n_grid_points = T_grid.shape[0] * K_grid.shape[0]
    
    results = {}
    
    # Adjust batch size if too large for data
    batch_size = min(args.batch_size, X_tr_n.shape[0])
    
    # ── 1. GPE Benchmark ─────────────────────────────────────────────────────
    if args.model in ["gpe", "all"]:
        print("\n--- Training GPE (Stochastic Variational GP)... ---")
        num_inducing = min(args.gp_inducing, X_tr_n.shape[0])
        inducing_idx = rng.choice(X_tr_n.shape[0], size=num_inducing, replace=False)
        inducing_points = torch.tensor(X_tr_n[inducing_idx], dtype=torch.float32)
        inducing_points = inducing_points.unsqueeze(0).expand(n_grid_points, -1, -1).to(device)
        
        gp_model = BatchSVGPModel(inducing_points).to(device)
        gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_grid_points])).to(device)
        gp_mll = gpytorch.mlls.VariationalELBO(gp_likelihood, gp_model, num_data=X_tr_n.shape[0]).to(device)
        
        gp_optimizer = torch.optim.Adam([
            {'params': gp_model.parameters()},
            {'params': gp_likelihood.parameters()}
        ], lr=args.gp_lr)
        
        tr_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32),
                              torch.tensor(Y_tr_n.reshape(-1, n_grid_points), dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        
        gp_model.train()
        gp_likelihood.train()
        
        t0_train = time.time()
        with gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), \
             gpytorch.settings.cholesky_jitter(double_value=1e-5, float_value=1e-3), \
             gpytorch.settings.skip_posterior_variances(True):
            for epoch in range(args.gp_epochs):
                epoch_loss = 0.0
                for X_b, Y_b in tr_loader:
                    X_b = X_b.to(device)
                    Y_b = Y_b.to(device)
                    
                    # Expand to GP batch format
                    X_gp = X_b.unsqueeze(0).expand(n_grid_points, -1, -1)
                    Y_gp = Y_b.t()
                    
                    gp_optimizer.zero_grad()
                    gp_out = gp_model(X_gp)
                    loss = -gp_mll(gp_out, Y_gp).sum()
                    loss.backward()
                    gp_optimizer.step()
                    epoch_loss += loss.item() * X_b.size(0)
                epoch_loss /= X_tr_n.shape[0]
                if (epoch + 1) % max(1, args.gp_epochs // 5) == 0 or args.smoke:
                    print(f"  GPE Epoch {epoch+1:02d}/{args.gp_epochs:02d} - Loss: {epoch_loss:.4f}")
        gp_train_time = time.time() - t0_train
        
        # GPE Inference
        gp_model.eval()
        gp_likelihood.eval()
        val_preds_list = []
        
        val_ds = TensorDataset(torch.tensor(X_va_n, dtype=torch.float32))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        t0_inf = time.time()
        with torch.no_grad(), \
             gpytorch.settings.fast_pred_var(), \
             gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), \
             gpytorch.settings.cholesky_jitter(double_value=1e-5, float_value=1e-3), \
             gpytorch.settings.skip_posterior_variances(True):
            for X_b, in val_loader:
                X_b = X_b.to(device)
                X_gp = X_b.unsqueeze(0).expand(n_grid_points, -1, -1)
                preds = gp_likelihood(gp_model(X_gp))
                val_preds_list.append(preds.mean.cpu().numpy())
        gp_inf_time = time.time() - t0_inf
        
        # Reconstruct prediction matrix: shape (N_val, nT, nK)
        Y_pred_gp_n = np.concatenate(val_preds_list, axis=1).T
        Y_pred_gp_n = Y_pred_gp_n.reshape(-1, T_grid.shape[0], K_grid.shape[0])
        Y_pred_gp = iv_normalizer.inverse_transform(Y_pred_gp_n)
        
        metrics_n = compute_metrics(Y_pred_gp_n, Y_va_n)
        metrics_raw = compute_metrics(Y_pred_gp, Y_va)
        
        results["GPE"] = {
            "train_time": gp_train_time,
            "inf_time": gp_inf_time,
            "norm_MSE": metrics_n["MSE"],
            "norm_MAE": metrics_n["MAE"],
            "norm_R2": metrics_n["R2"],
            "raw_MSE": metrics_raw["MSE"],
            "raw_MAE": metrics_raw["MAE"],
            "raw_R2": metrics_raw["R2"]
        }
        
        # delete GPE model, optimizer, likelihood, tensors
        del gp_model, gp_optimizer, gp_likelihood, gp_mll
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    # ── 2. PCE Benchmark ─────────────────────────────────────────────────────
    if args.model in ["pce", "all"]:
        print("\n--- Training PCE (Polynomial Chaos Expansion)... ---")
        pce_dist = cp.Iid(cp.Uniform(-np.sqrt(3), np.sqrt(3)), n_params)
        pce_expansion = cp.generate_expansion(args.pce_order, pce_dist)
        
        Y_tr_n_flat = Y_tr_n.reshape(-1, n_grid_points)
        
        pce_cache_path = os.path.join(cache_dir, f"pce_design_N{N}_seed{args.seed}_order{args.pce_order}.npz")
        
        t0_train = time.time()
        if os.path.exists(pce_cache_path):
            print(f"Loading cached PCE design matrices from {pce_cache_path}")
            pce_data = np.load(pce_cache_path)
            design_matrix_tr = pce_data["design_matrix_tr"]
            design_matrix_va = pce_data["design_matrix_va"]
        else:
            print("Evaluating ChaosPy for training PCE design matrix...")
            X_tr_n_cp = X_tr_n.astype(np.float64).T
            design_matrix_tr = pce_expansion(*X_tr_n_cp).T
        
        # Decide regression method
        reg_type = args.pce_reg
        if reg_type == 'auto':
            if X_tr_n.shape[0] >= 10000:
                reg_type = 'ols'
            else:
                reg_type = 'ridge'
                
        print(f"  PCE Regression Type: {reg_type.upper()}")
        
        cv_folds = min(3, X_tr_n.shape[0])
        is_underdetermined = X_tr_n.shape[0] < len(pce_expansion)
        use_cv = (not is_underdetermined) and (cv_folds >= 2)
        
        if reg_type == 'ols':
            # Analytical OLS via np.linalg.lstsq
            coef = np.linalg.lstsq(design_matrix_tr, Y_tr_n_flat, rcond=None)[0]
            predict_fn = lambda dm: dm @ coef
        elif reg_type == 'ridge':
            if not use_cv:
                print(f"  Warning: training samples ({X_tr_n.shape[0]}) < PCE terms ({len(pce_expansion)}) or cv_folds ({cv_folds}) < 2. Falling back to Ridge without CV.")
                ridge_model = Ridge(alpha=1.0)
            else:
                ridge_model = RidgeCV(cv=cv_folds)
            ridge_model.fit(design_matrix_tr, Y_tr_n_flat)
            predict_fn = lambda dm: ridge_model.predict(dm)
        elif reg_type == 'lasso':
            if not use_cv:
                print(f"  Warning: training samples ({X_tr_n.shape[0]}) < PCE terms ({len(pce_expansion)}) or cv_folds ({cv_folds}) < 2. Falling back to Ridge without CV.")
                fallback_model = Ridge(alpha=1.0)
                fallback_model.fit(design_matrix_tr, Y_tr_n_flat)
                predict_fn = lambda dm: fallback_model.predict(dm)
            else:
                lasso_model = MultiTaskLassoCV(cv=cv_folds, max_iter=2000)
                lasso_model.fit(design_matrix_tr, Y_tr_n_flat)
                predict_fn = lambda dm: lasso_model.predict(dm)
        else:
            raise ValueError(f"Unsupported PCE regression method: {reg_type}")
            
        pce_train_time = time.time() - t0_train
        
        # PCE Inference
        t0_inf = time.time()
        if not os.path.exists(pce_cache_path):
            print("Evaluating ChaosPy for validation PCE design matrix...")
            X_va_n_cp = X_va_n.astype(np.float64).T
            design_matrix_va = pce_expansion(*X_va_n_cp).T
            np.savez(pce_cache_path, design_matrix_tr=design_matrix_tr, design_matrix_va=design_matrix_va)
            
        Y_pred_pce_n_flat = predict_fn(design_matrix_va)
        pce_inf_time = time.time() - t0_inf
        
        Y_pred_pce_n = Y_pred_pce_n_flat.reshape(-1, T_grid.shape[0], K_grid.shape[0])
        Y_pred_pce = iv_normalizer.inverse_transform(Y_pred_pce_n)
        
        metrics_n = compute_metrics(Y_pred_pce_n, Y_va_n)
        metrics_raw = compute_metrics(Y_pred_pce, Y_va)
        
        results["PCE"] = {
            "train_time": pce_train_time,
            "inf_time": pce_inf_time,
            "norm_MSE": metrics_n["MSE"],
            "norm_MAE": metrics_n["MAE"],
            "norm_R2": metrics_n["R2"],
            "raw_MSE": metrics_raw["MSE"],
            "raw_MAE": metrics_raw["MAE"],
            "raw_R2": metrics_raw["R2"]
        }
        
    # ── 3. FNO Benchmark ─────────────────────────────────────────────────────
    if args.model in ["fno", "all"]:
        print("\n--- Training FNO (Fourier Neural Operator)... ---")
        fno_model = MirrorPaddedFNO2d(param_dim=n_params).to(device)
        fno_optimizer = torch.optim.AdamW(fno_model.parameters(), lr=args.lr, weight_decay=1e-4)
        fno_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fno_optimizer, T_max=args.epochs)
        
        fno_spatial = _make_spatial_input(T_grid, K_grid, device)
        t_grid_tensor = torch.tensor(T_grid, dtype=torch.float32, device=device)
        k_grid_tensor = torch.tensor(K_grid, dtype=torch.float32, device=device)
        
        tr_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32),
                              torch.tensor(Y_tr_n, dtype=torch.float32))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        
        mean_t = torch.tensor(iv_normalizer.mean, dtype=torch.float32, device=device)
        std_t = torch.tensor(iv_normalizer.std, dtype=torch.float32, device=device)
        
        t0_train = time.time()
        for epoch in range(args.epochs):
            fno_model.train()
            epoch_loss = 0.0
            for X_b, Y_b in tr_loader:
                X_b = X_b.to(device)
                Y_b = Y_b.to(device)
                B = X_b.size(0)
                
                sp = fno_spatial.expand(B, -1, -1, -1)
                pred = fno_model(sp, X_b)
                
                loss_huber = F.huber_loss(pred, Y_b, delta=1.0)
                pred_denorm = pred * std_t + mean_t
                loss_arb = arbitrage_free_regularization(pred_denorm, t_grid_tensor, k_grid_tensor)
                loss_neg = F.relu(-pred_denorm).mean()
                
                loss = loss_huber + 1e-4 * loss_arb + 1.0 * loss_neg
                
                fno_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fno_model.parameters(), 1.0)
                fno_optimizer.step()
                epoch_loss += loss.item() * B
                
            fno_scheduler.step()
            epoch_loss /= X_tr_n.shape[0]
            if (epoch + 1) % max(1, args.epochs // 5) == 0 or args.smoke:
                print(f"  FNO Epoch {epoch+1:02d}/{args.epochs:02d} - Loss: {epoch_loss:.4f}")
        fno_train_time = time.time() - t0_train
        
        # FNO Inference
        fno_model.eval()
        val_preds_list = []
        val_ds = TensorDataset(torch.tensor(X_va_n, dtype=torch.float32))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        t0_inf = time.time()
        with torch.no_grad():
            for X_b, in val_loader:
                X_b = X_b.to(device)
                B = X_b.size(0)
                sp = fno_spatial.expand(B, -1, -1, -1)
                pred = fno_model(sp, X_b)
                val_preds_list.append(pred.cpu().numpy())
        fno_inf_time = time.time() - t0_inf
        
        Y_pred_fno_n = np.concatenate(val_preds_list, axis=0)
        Y_pred_fno = iv_normalizer.inverse_transform(Y_pred_fno_n)
        
        metrics_n = compute_metrics(Y_pred_fno_n, Y_va_n)
        metrics_raw = compute_metrics(Y_pred_fno, Y_va)
        
        results["FNO"] = {
            "train_time": fno_train_time,
            "inf_time": fno_inf_time,
            "norm_MSE": metrics_n["MSE"],
            "norm_MAE": metrics_n["MAE"],
            "norm_R2": metrics_n["R2"],
            "raw_MSE": metrics_raw["MSE"],
            "raw_MAE": metrics_raw["MAE"],
            "raw_R2": metrics_raw["R2"]
        }
        
    # ── Print Benchmarking Results ───────────────────────────────────────────
    print("\n" + "="*80)
    print(f"  BENCHMARK RESULTS (N = {N:,}, Train Split = 80%)")
    print("="*80)
    print(f"{'Model':<8} | {'Train (s)':<10} | {'Inference (s)':<13} | {'Norm R2':<10} | {'Raw MSE':<12} | {'Raw R2':<10}")
    print("-"*80)
    for model_name, metrics in results.items():
        print(f"{model_name:<8} | {metrics['train_time']:<10.4f} | {metrics['inf_time']:<13.4f} | "
              f"{metrics['norm_R2']:<10.4f} | {metrics['raw_MSE']:<12.6f} | {metrics['raw_R2']:<10.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
