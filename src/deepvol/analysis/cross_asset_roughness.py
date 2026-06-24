"""
cross_asset_roughness.py

Optimized implementation of src/deepvol/analysis/cross_asset_roughness.py
for Phase 9 Milestone M1: Cross-Asset Roughness Study.
"""

from __future__ import annotations
import gc
import os
import sys
import json
import hashlib
import warnings
import threading
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Ensure parent and src are in path
project_root = Path(__file__).resolve().parents[3]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# KNOWN LIMITATION (F-12): PyTorch's torch.vmap / torch.func.jacfwd cannot trace
# through the original SpectralConv2d.forward because it uses in-place indexing on
# a zero-initialized tensor (out_ft[:, :, :modes1, :modes2] = ...), which is not
# supported by PyTorch's functional transforms (functorch).
# This patched version replaces the in-place scatter with F.pad + addition, making
# the forward pass purely functional and vmap-compatible.
# TODO: Remove this patch once PyTorch natively supports in-place ops in vmap
# (tracked in pytorch/functorch#667).
from deepvol.surrogates import fno_model
import torch.nn.functional as F

def _spectral_conv2d_forward_patched(self, x):
    """vmap-compatible SpectralConv2d forward pass using pad+add instead of in-place scatter."""
    B = x.shape[0]
    x_ft = torch.fft.rfft2(x)
    H, W = x.size(-2), x.size(-1)//2+1
    
    w1_part = torch.einsum(
        "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
    w2_part = torch.einsum(
        "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
    w1_padded = F.pad(w1_part, (0, W - self.modes2, 0, H - self.modes1))
    w2_padded = F.pad(w2_part, (0, W - self.modes2, H - self.modes1, 0))
    
    out_ft = w1_padded + w2_padded
    return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

fno_model.SpectralConv2d.forward = _spectral_conv2d_forward_patched



from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.calibration import calibrate_bfgs as _cal_mod
from deepvol.calibration.batch_calibration import CalibrationResult, results_to_dataframe
from deepvol.market.spx_data import clean_chain, to_iv_surface
from deepvol.analysis.crypto_hurst import align_crypto_inputs
from deepvol.market.deribit_data import build_iv_surface


class ParameterTrajectoryGenerator:
    """
    Optimized Parameter Trajectory Generator. Pre-computes and caches the entire 4-year
    parameter trajectory using discretized Ornstein-Uhlenbeck processes.
    """
    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
        self.dates = pd.bdate_range(start=start_date, end=end_date).strftime("%Y-%m-%d").tolist()
        self.base_params = {
            "SPX":    {"kappa": 1.0, "theta": 0.04,  "sigma": 0.50, "rho": -0.75, "v0": 0.04,  "H": 0.06},
            "BTC":    {"kappa": 1.0, "theta": 0.12,  "sigma": 0.60, "rho": -0.45, "v0": 0.10,  "H": 0.10},
            "ETH":    {"kappa": 1.0, "theta": 0.14,  "sigma": 0.70, "rho": -0.50, "v0": 0.12,  "H": 0.09},
            "EURUSD": {"kappa": 1.0, "theta": 0.015, "sigma": 0.30, "rho": -0.15, "v0": 0.015, "H": 0.12},
            "WTI":    {"kappa": 1.0, "theta": 0.08,  "sigma": 0.50, "rho": -0.35, "v0": 0.08,  "H": 0.11},
        }
        self._cache: Dict[str, pd.DataFrame] = {}
        self._precompute_all()

    def _precompute_all(self):
        """Pre-computes parameter trajectories for all assets via discretized OU processes."""
        n_days = len(self.dates)
        dt = 1 / 252.0
        
        # Datetime objects for shock masks
        dt_objs = np.array([datetime.strptime(d, "%Y-%m-%d").date() for d in self.dates])
        
        # Covid shock factor mask
        covid_mask = (dt_objs >= date(2020, 3, 1)) & (dt_objs <= date(2020, 4, 30))
        covid_peak = date(2020, 3, 16)
        covid_dists = np.array([abs((d - covid_peak).days) for d in dt_objs])
        covid_factors = np.clip(3.5 - 0.1 * covid_dists, 1.0, None)
        
        # Crypto bull market mask (May 2021)
        crypto_mask = (dt_objs >= date(2021, 5, 1)) & (dt_objs <= date(2021, 5, 31))
        
        # Oil crash mask (April 15 - 25, 2020)
        oil_mask = (dt_objs >= date(2020, 4, 15)) & (dt_objs <= date(2020, 4, 25))

        for asset, base in self.base_params.items():
            # Seed PRNG based on asset name for deterministic daily noise
            asset_seed = int(hashlib.md5(asset.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(asset_seed)
            
            # Generate daily random noise for all days
            v0_eps = rng.normal(0, 1.0, n_days)
            H_eps = rng.normal(0, 1.0, n_days)
            rho_eps = rng.normal(0, 1.0, n_days)
            sigma_eps = rng.normal(0, 1.0, n_days)
            
            v0 = np.zeros(n_days)
            H = np.zeros(n_days)
            rho = np.zeros(n_days)
            sigma = np.zeros(n_days)
            
            v0[0] = base["v0"]
            H[0] = base["H"]
            rho[0] = base["rho"]
            sigma[0] = base["sigma"]
            
            v0_eta = 2.0
            v0_xi = 0.15
            
            H_eta = 0.5
            H_xi = 0.02
            
            other_eta = 0.5
            other_xi = 0.05
            
            v0_factor = np.exp(-v0_eta * dt)
            H_factor = np.exp(-H_eta * dt)
            other_factor = np.exp(-other_eta * dt)
            
            v0_noise_std = v0_xi * np.sqrt(dt)
            H_noise_std = H_xi * np.sqrt(dt)
            other_noise_std = other_xi * np.sqrt(dt)
            
            for t in range(1, n_days):
                v0[t] = base["v0"] + (v0[t-1] - base["v0"]) * v0_factor + v0_noise_std * v0_eps[t]
                H[t] = base["H"] + (H[t-1] - base["H"]) * H_factor + H_noise_std * H_eps[t]
                rho[t] = base["rho"] + (rho[t-1] - base["rho"]) * other_factor + other_noise_std * rho_eps[t]
                sigma[t] = base["sigma"] + (sigma[t-1] - base["sigma"]) * other_factor + other_noise_std * sigma_eps[t]
            
            theta = np.full(n_days, base["theta"])
            kappa = np.full(n_days, base["kappa"])
            
            # Apply historical shocks
            # 1. COVID-19 Shock
            v0 = np.where(covid_mask, v0 * covid_factors, v0)
            theta = np.where(covid_mask, theta * covid_factors, theta)
            H = np.where(covid_mask, np.clip(H - 0.04 * (covid_factors - 1.0) / 2.5, 0.04, None), H)
            rho = np.where(covid_mask, np.clip(rho - 0.1 * (covid_factors - 1.0) / 2.5, -0.90, None), rho)
            
            # 2. Crypto Spring
            if asset in ("BTC", "ETH"):
                v0 = np.where(crypto_mask, v0 * 2.0, v0)
                theta = np.where(crypto_mask, theta * 1.8, theta)
                
            # 3. Negative Oil
            if asset == "WTI":
                v0 = np.where(oil_mask, v0 * 3.0, v0)
                rho = np.where(oil_mask, -0.10, rho)
                
            # Clamp to FNO bounds
            v0 = np.clip(v0, 0.01, 0.15)
            theta = np.clip(theta, 0.01, 0.15)
            sigma = np.clip(sigma, 0.1, 1.0)
            rho = np.clip(rho, -0.9, -0.1)
            H = np.clip(H, 0.04, 0.15)
            
            df = pd.DataFrame({
                "kappa": kappa,
                "theta": theta,
                "sigma": sigma,
                "rho": rho,
                "v0": v0,
                "H": H
            }, index=self.dates)
            
            self._cache[asset] = df

    def get_parameters(self, asset: str, date_str: str) -> Dict[str, float]:
        """O(1) parameter lookup from pre-computed trajectory cache."""
        asset_upper = asset.upper().replace("/", "").replace(" ", "").replace("_", "")
        if "WTI" in asset_upper:
            asset_upper = "WTI"
        elif "EURUSD" in asset_upper:
            asset_upper = "EURUSD"

        if asset_upper not in self._cache:
            raise ValueError(f"Unknown asset: {asset}")
        df = self._cache[asset_upper]
        if date_str not in df.index:
            # Fallback to closest available date
            return df.iloc[len(df)//2].to_dict()
        return df.loc[date_str].to_dict()


class CrossAssetDataPipeline:
    """
    Manages option implied volatility surface loading and generation for 2020-2023.
    """
    def __init__(self, project_root_path: Path, trajectory_gen: ParameterTrajectoryGenerator):
        self.project_root = project_root_path
        self.tg = trajectory_gen
        self.t_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
        self.k_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
        pn_path = project_root_path / "artifacts" / "models" / "param_normalizer_v3.npz"
        yn_path = project_root_path / "artifacts" / "models" / "iv_normalizer_v3.npz"
        if not pn_path.exists():
            pn_path = Path(__file__).parents[3] / "artifacts" / "models" / "param_normalizer_v3.npz"
        if not yn_path.exists():
            yn_path = Path(__file__).parents[3] / "artifacts" / "models" / "iv_normalizer_v3.npz"
        self.pn = ParameterNormalizer.load(str(pn_path))
        self.yn = IVSurfaceNormalizer.load(str(yn_path))

    def get_surface(self, asset: str, date_str: str, model: torch.nn.Module, device: torch.device) -> np.ndarray:
        """
        Retrieves options implied volatility surface. Loads cache if exists, or generates synthetically.
        """
        asset_upper = asset.upper().replace("/", "").replace(" ", "").replace("_", "")
        if "WTI" in asset_upper:
            asset_upper = "WTI"
        elif "EURUSD" in asset_upper:
            asset_upper = "EURUSD"

        # Check cache folders
        cache_file = None
        market_dir = Path("/home/execorn/programming/derivatives/data/market")
        if not market_dir.exists():
            market_dir = self.project_root / "data" / "market"

        if asset_upper == "SPX":
            cache_file = market_dir / "spx" / f"spx_chain_{date_str}.parquet"
        elif asset_upper in ("BTC", "ETH"):
            cache_file = market_dir / "deribit" / f"{asset_upper.lower()}_chain_{date_str}.parquet"

        # Load from cache if file exists and has data
        if cache_file and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if asset_upper == "SPX":
                    df_c = clean_chain(df)
                    return to_iv_surface(df_c, S=5000.0, r=0.05, q=0.015)
                else:
                    df = align_crypto_inputs(df)
                    return build_iv_surface(df, self.t_grid, self.k_grid)
            except Exception as e:
                warnings.warn(f"Failed to load cached surface for {asset_upper} on {date_str}: {e}. Falling back to synthetic.")

        # Generate synthetically using parameter trajectory and FNO v3
        params = self.tg.get_parameters(asset_upper, date_str)

        # Build parameter tensor
        theta_raw = torch.tensor(
            [[params["kappa"], params["theta"], params["sigma"], params["rho"], params["v0"], params["H"]]],
            dtype=torch.float32, device=device
        )

        import deepvol.calibration.calibrate_bfgs as calibrate_bfgs
        calibrate_bfgs._param_norm = self.pn
        calibrate_bfgs._iv_norm = self.yn
        from deepvol.calibration.calibrate_bfgs import _make_spatial_input, _fno_predict_real_iv
        spatial = _make_spatial_input(self.t_grid, self.k_grid, device)

        with torch.no_grad():
            iv_surface_t = _fno_predict_real_iv(model, theta_raw, spatial)

        surface = iv_surface_t.squeeze().cpu().numpy()

        # Add 50 bps of measurement noise to simulate market microstructure noise
        seed_str = f"noise_{asset_upper}_{date_str}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.005, surface.shape)

        return np.clip(surface + noise, 1e-4, 2.5)


def _reparam_to_6d_with_H(
    v0: torch.Tensor, zeta: torch.Tensor, lam: torch.Tensor, H: torch.Tensor, device,
    theta: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Back-transform (v₀, ζ, λ, H) → raw 6D parameter vector (B,6)."""
    sigma = torch.sqrt(zeta**2 + lam**2).clamp(min=0.01)
    rho   = (zeta / sigma).clamp(-0.9, -0.1)
    kappa = torch.full_like(v0, 1.0)
    if theta is None:
        theta = torch.full_like(v0, 0.08)
    else:
        if theta.shape != v0.shape:
            theta = theta.expand_as(v0)
    H_clp = H.clamp(0.04, 0.15)
    return torch.stack([kappa, theta, sigma, rho, v0, H_clp], dim=-1).to(device)


def _make_spatial(device: torch.device) -> torch.Tensor:
    """Build (1,8,11,2) spatial coordinate tensor."""
    T = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float32)
    K = torch.linspace(-0.5, 0.5, 11, dtype=torch.float32)
    T_norm = (T - T.mean()) / (T.std() + 1e-8)
    K_norm = K / 0.5
    T_m, K_m = torch.meshgrid(T_norm, K_norm, indexing="ij")
    return torch.stack([T_m, K_m], dim=-1).unsqueeze(0).to(device)


def calibrate_newton_h_batch(
    model, target_iv_batch, pn, yn, device,
    max_iter: int = 15, tol: float = 1e-6, eps_lm: float = 1e-4,
    prior_batch: Optional[torch.Tensor] = None,
    reg_weights: Optional[torch.Tensor] = None,
    true_theta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calibrate a batch of B target implied volatility surfaces concurrently on the GPU/CPU.
    Optimizes in 4D space (v0, zeta, lam, H) and back-transforms to 6D via _reparam_to_6d_with_H.
    Supports regularization towards a prior parameter vector.
    """
    B = target_iv_batch.shape[0]
    target_flat = target_iv_batch.reshape(B, 88)
    
    spatial_single = _make_spatial(device).squeeze(0)
    
    # 4D bounds: [v0, zeta, lam, H]
    lo_t = torch.tensor([0.01, -0.90, 0.01, 0.04], dtype=torch.float32, device=device)
    hi_t = torch.tensor([0.15, -0.01, 0.99, 0.15], dtype=torch.float32, device=device)
    
    pn_mean = torch.tensor(pn.mean, dtype=torch.float32, device=device)
    pn_std = torch.tensor(pn.std, dtype=torch.float32, device=device)
    yn_mean = torch.tensor(yn.mean, dtype=torch.float32, device=device)
    yn_std = torch.tensor(yn.std, dtype=torch.float32, device=device)

    # 4 starting points in 4D space (v0, zeta, lam, H)
    inits = torch.tensor([
        [0.08, -0.25, 0.4330127, 0.08],
        [0.04, -0.21, 0.2142428, 0.06],
        [0.12, -0.21, 0.6677574, 0.10],
        [0.04, -0.32, 0.7332121, 0.12]
    ], dtype=torch.float32, device=device)
    
    num_starts = len(inits)
    theta = inits.repeat_interleave(B, dim=0)       # (M=B*num_starts, 4)
    M = B * num_starts
    spatial_batch = spatial_single.unsqueeze(0).repeat(M, 1, 1, 1)

    if true_theta is None:
        true_theta = torch.full((B,), 0.08, dtype=torch.float32, device=device)
    true_theta_expanded = true_theta.repeat(num_starts)

    if prior_batch is not None:
        assert reg_weights is not None, "reg_weights must be provided if prior_batch is provided"
        
        def fwd_fn(theta_single, spatial_single, prior_single, theta_val_single):
            v0, zeta, lam, H = theta_single[0], theta_single[1], theta_single[2], theta_single[3]
            p6 = _reparam_to_6d_with_H(
                v0.unsqueeze(0), zeta.unsqueeze(0), lam.unsqueeze(0), H.unsqueeze(0), device,
                theta=theta_val_single.unsqueeze(0)
            ).squeeze(0)
            theta_norm = (p6.unsqueeze(0) - pn_mean) / pn_std
            theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
            spatial_input = spatial_single.unsqueeze(0)
            pred = model(spatial_input, theta_norm)
            iv = pred * yn_std + yn_mean
            iv_flat = iv.clamp(min=1e-4).reshape(-1)
            reg_term = reg_weights * (theta_single - prior_single)
            return torch.cat([iv_flat, reg_term], dim=0)

        vmap_fwd = torch.vmap(fwd_fn, in_dims=(0, 0, 0, 0))
        vmap_jac = torch.vmap(torch.func.jacfwd(fwd_fn, argnums=0), in_dims=(0, 0, 0, 0))
        
        target_augmented = torch.cat([target_flat, torch.zeros(B, 4, device=device)], dim=1)
        target_expanded = target_augmented.repeat(num_starts, 1)
        prior_expanded = prior_batch.repeat(num_starts, 1)
    else:
        def fwd_fn(theta_single, spatial_single, theta_val_single):
            v0, zeta, lam, H = theta_single[0], theta_single[1], theta_single[2], theta_single[3]
            p6 = _reparam_to_6d_with_H(
                v0.unsqueeze(0), zeta.unsqueeze(0), lam.unsqueeze(0), H.unsqueeze(0), device,
                theta=theta_val_single.unsqueeze(0)
            ).squeeze(0)
            theta_norm = (p6.unsqueeze(0) - pn_mean) / pn_std
            theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
            spatial_input = spatial_single.unsqueeze(0)
            pred = model(spatial_input, theta_norm)
            iv = pred * yn_std + yn_mean
            return iv.clamp(min=1e-4).reshape(-1)

        vmap_fwd = torch.vmap(fwd_fn, in_dims=(0, 0, 0))
        vmap_jac = torch.vmap(torch.func.jacfwd(fwd_fn, argnums=0), in_dims=(0, 0, 0))
        
        target_expanded = target_flat.repeat(num_starts, 1)

    # Dynamically select chunk size based on device to maximize GPU utilization
    chunk_sz = 4
    
    loss_best = torch.full((M,), float('inf'), device=device)

    for it in range(max_iter):
        preds = []
        jacs = []
        for i in range(0, M, chunk_sz):
            theta_sub = theta[i:i+chunk_sz]
            spatial_sub = spatial_batch[i:i+chunk_sz]
            theta_val_sub = true_theta_expanded[i:i+chunk_sz]
            if prior_batch is not None:
                prior_sub = prior_expanded[i:i+chunk_sz]
                preds.append(vmap_fwd(theta_sub, spatial_sub, prior_sub, theta_val_sub).detach())
                jacs.append(vmap_jac(theta_sub, spatial_sub, prior_sub, theta_val_sub).detach())
            else:
                preds.append(vmap_fwd(theta_sub, spatial_sub, theta_val_sub).detach())
                jacs.append(vmap_jac(theta_sub, spatial_sub, theta_val_sub).detach())
            
        pred_val = torch.cat(preds, dim=0)
        jac_val = torch.cat(jacs, dim=0)
        
        r = pred_val - target_expanded
        loss = (r**2).mean(dim=1)
            
        # Solve LM equations: (J^T J + epsilon * diag(J^T J)) delta = -J^T r
        JtJ = torch.bmm(jac_val.transpose(1, 2), jac_val)
        Jtr = torch.bmm(jac_val.transpose(1, 2), r.unsqueeze(-1))
        
        diag_JtJ = torch.diagonal(JtJ, dim1=1, dim2=2)
        # diag_JtJ shape (M, 4)
        # index 3 is H
        h_damp = torch.zeros_like(diag_JtJ)
        h_damp[:, 3] = 1.0
        eps = eps_lm * diag_JtJ.clamp(min=1e-8) + 1e-9 + h_damp * 0.1  # 0.1 extra dampening for H
        JtJ_reg = JtJ + torch.diag_embed(eps)
        
        delta = torch.linalg.solve(JtJ_reg, -Jtr).squeeze(-1)
        
        # Backtracking line search on GPU
        theta_best = theta.clone()
        loss_best = torch.where(loss_best == float('inf'), loss, loss_best)
        alpha = torch.ones(M, 1, device=device)
        
        for ls_step in range(4):
            theta_cand = (theta + alpha * delta).clamp(lo_t + 1e-5, hi_t - 1e-5)
            preds_cand = []
            for i in range(0, M, chunk_sz):
                theta_sub = theta_cand[i:i+chunk_sz]
                spatial_sub = spatial_batch[i:i+chunk_sz]
                theta_val_sub = true_theta_expanded[i:i+chunk_sz]
                with torch.no_grad():
                    if prior_batch is not None:
                        prior_sub = prior_expanded[i:i+chunk_sz]
                        preds_cand.append(vmap_fwd(theta_sub, spatial_sub, prior_sub, theta_val_sub).detach())
                    else:
                        preds_cand.append(vmap_fwd(theta_sub, spatial_sub, theta_val_sub).detach())
            pred_cand_val = torch.cat(preds_cand, dim=0)
            loss_cand = ((pred_cand_val - target_expanded) ** 2).mean(dim=1)
            
            better = loss_cand < loss_best
            theta_best = torch.where(better.unsqueeze(-1), theta_cand, theta_best)
            loss_best = torch.where(better, loss_cand, loss_best)
            alpha = alpha * 0.5
            
        theta = theta_best.detach()
        
    loss_reshaped = loss_best.reshape(num_starts, B)
    best_start_idx = loss_reshaped.argmin(dim=0)
    
    theta_reshaped = theta.reshape(num_starts, B, 4)
    best_theta_4d = theta_reshaped[best_start_idx, torch.arange(B, device=device)]
    
    # Back-transform best 4D theta to 6D
    best_theta_6d = _reparam_to_6d_with_H(
        best_theta_4d[:, 0], best_theta_4d[:, 1], best_theta_4d[:, 2], best_theta_4d[:, 3], device,
        theta=true_theta
    )
    
    def fwd_fn_6d(theta_6d_single, spatial_single):
        theta_norm = (theta_6d_single.unsqueeze(0) - pn_mean) / pn_std
        theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
        spatial_input = spatial_single.unsqueeze(0)
        pred = model(spatial_input, theta_norm)
        iv = pred * yn_std + yn_mean
        return iv.clamp(min=1e-4).reshape(-1)

    vmap_fwd_6d = torch.vmap(fwd_fn_6d, in_dims=(0, 0))

    best_spatial = spatial_single.unsqueeze(0).repeat(B, 1, 1, 1)
    best_preds = []
    for i in range(0, B, chunk_sz):
        theta_sub = best_theta_6d[i:i+chunk_sz]
        spatial_sub = best_spatial[i:i+chunk_sz]
        with torch.no_grad():
            best_preds.append(vmap_fwd_6d(theta_sub, spatial_sub).detach())
    final_preds = torch.cat(best_preds, dim=0).reshape(B, 8, 11)
    
    final_loss = ((final_preds.reshape(B, 88) - target_flat) ** 2).mean(dim=1)
    return best_theta_6d, final_preds, final_loss


def calibrate_batch_h(
    dates: List[str],
    currency: str,
    target_surfaces: Dict[str, np.ndarray],
    model,
    pn,
    yn,
    device: torch.device,
    prior_batch: Optional[torch.Tensor] = None,
    reg_weights: Optional[torch.Tensor] = None,
    tg: Optional[ParameterTrajectoryGenerator] = None,
) -> List[CalibrationResult]:
    """
    Calibrate 4D H-parameter space for a batch of dates.
    """
    total = len(dates)
    if total == 0:
        return []
        
    target_surfaces_list = [target_surfaces[d] for d in dates]
    target_iv_batch = np.stack(target_surfaces_list, axis=0) # (B, 8, 11)
    target_iv_tensor = torch.tensor(target_iv_batch, dtype=torch.float32, device=device)
    
    if reg_weights is not None and not isinstance(reg_weights, torch.Tensor):
        reg_weights = torch.tensor(reg_weights, dtype=torch.float32, device=device)
        
    if tg is not None:
        true_thetas = [tg.get_parameters(currency, d)["theta"] for d in dates]
        true_theta_tensor = torch.tensor(true_thetas, dtype=torch.float32, device=device)
    else:
        true_theta_tensor = None

    import time
    t_start = time.perf_counter()
    cal_theta, cal_preds, cal_loss = calibrate_newton_h_batch(
        model, target_iv_tensor, pn, yn, device,
        prior_batch=prior_batch, reg_weights=reg_weights,
        true_theta=true_theta_tensor
    )
    t_end = time.perf_counter()
    
    t_total_ms = (t_end - t_start) * 1000.0
    runtime_ms_per_surface = t_total_ms / total
    
    results = []
    cal_theta_np = cal_theta.cpu().numpy()
    cal_preds_np = cal_preds.cpu().numpy()
    cal_loss_np = cal_loss.cpu().numpy()
    
    PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]
    for i, d in enumerate(dates):
        rmse = float(np.sqrt(cal_loss_np[i]) * 10000.0)
        converged = bool(rmse < 100.0)  # Convergence threshold: 100 bps
        
        result = CalibrationResult(
            date=d,
            currency=currency.upper(),
            params={n: float(v) for n, v in zip(PARAM_NAMES, cal_theta_np[i].tolist())},
            rmse_bps=rmse,
            runtime_ms=runtime_ms_per_surface,
            converged=converged,
            surface=cal_preds_np[i],
        )
        results.append(result)
        
    return sorted(results, key=lambda r: r.date)


def save_results(results: List[CalibrationResult], path: str) -> None:
    """
    Serialize a list of CalibrationResults to a JSON file, omitting the surface.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        d = {
            "date": r.date,
            "currency": r.currency,
            "params": r.params,
            "rmse_bps": r.rmse_bps,
            "runtime_ms": r.runtime_ms,
            "converged": r.converged,
            "surface": None
        }
        data.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_results(path: str) -> List[CalibrationResult]:
    """
    Load CalibrationResults from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CalibrationResult.from_dict(d) for d in data]


def run_cross_asset_study(
    start: str,
    end: str,
    assets: List[str],
    project_root_dir: Optional[str] = None,
    max_workers: int = 4,
    device: str = "cpu",
    batch_size: int = 128,
    use_compile: bool = False,
    reg_weights: List[float] = [0.01, 0.05, 0.05, 0.15]
) -> Dict[str, pd.DataFrame]:
    """
    Main study orchestrator. Performs batched calibrations under a GPU Lock.
    """
    if project_root_dir is None:
        project_root_dir = str(Path(__file__).resolve().parents[3])

    root_path = Path(project_root_dir)
    results_dir = root_path / "results" / "cross_asset"
    results_dir.mkdir(parents=True, exist_ok=True)

    gpu_lock = threading.Lock()
    dev = torch.device(device)

    # Load normalizers
    from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
    
    pn_path = root_path / "artifacts" / "models" / "param_normalizer_v3.npz"
    yn_path = root_path / "artifacts" / "models" / "iv_normalizer_v3.npz"
    
    if not pn_path.exists():
        pn_path = Path(__file__).parents[3] / "artifacts" / "models" / "param_normalizer_v3.npz"
    if not yn_path.exists():
        yn_path = Path(__file__).parents[3] / "artifacts" / "models" / "iv_normalizer_v3.npz"

    pn = ParameterNormalizer.load(str(pn_path))
    yn = IVSurfaceNormalizer.load(str(yn_path))

    import deepvol.calibration.calibrate_bfgs as calibrate_bfgs
    calibrate_bfgs._param_norm = pn
    calibrate_bfgs._iv_norm = yn

    weights_path = root_path / "artifacts" / "weights" / "fno_v3_final_prod.pth"
    model = MirrorPaddedFNO2d(param_dim=6)
    model.load_state_dict(torch.load(weights_path, map_location=dev, weights_only=True))
    model.to(dev).eval()

    if use_compile and hasattr(torch, "compile"):
        try:
            print("JIT Compiling FNO surrogate model via torch.compile...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            warnings.warn(f"Failed to JIT compile model: {e}. Falling back to eager mode.")

    all_dates = pd.bdate_range(start=start, end=end).strftime("%Y-%m-%d").tolist()

    tg = ParameterTrajectoryGenerator(start_date=start, end_date=end)
    pipeline = CrossAssetDataPipeline(root_path, tg)

    study_results = {}

    for asset in assets:
        asset_upper = asset.upper().replace("/", "").replace(" ", "").replace("_", "")
        if "WTI" in asset_upper:
            asset_upper = "WTI"
        elif "EURUSD" in asset_upper:
            asset_upper = "EURUSD"

        file_path = results_dir / f"{asset_upper}_hurst_study.json"

        # Load existing results to support resume
        existing_results = []
        if file_path.exists():
            try:
                existing_results = load_results(str(file_path))
                print(f"Loaded {len(existing_results)} existing results for {asset_upper} from {file_path}")
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}. Starting fresh.")

        completed_dates = {r.date for r in existing_results}
        missing_dates = [d for d in all_dates if d not in completed_dates]

        if missing_dates:
            print(f"Calibrating {len(missing_dates)} missing dates for {asset_upper}...")

            for idx in range(0, len(missing_dates), batch_size):
                chunk = missing_dates[idx : idx + batch_size]

                # Fetch surfaces on CPU
                target_surfaces = {}
                for d in chunk:
                    target_surfaces[d] = pipeline.get_surface(asset_upper, d, model, dev)

                # Construct priors based on asset's base parameters
                base = tg.base_params[asset_upper]
                zeta_base = base["sigma"] * base["rho"]
                lam_base = base["sigma"] * np.sqrt(1 - base["rho"]**2)
                prior_4d = [base["v0"], zeta_base, lam_base, base["H"]]
                prior_chunk = np.tile(prior_4d, (len(chunk), 1))
                prior_chunk_tensor = torch.tensor(prior_chunk, dtype=torch.float32, device=dev)

                # Calibrate chunk (GPU lock)
                print(f"Calibrating batch {idx//batch_size + 1} for {asset_upper} (size {len(chunk)})...")
                with gpu_lock:
                    chunk_results = calibrate_batch_h(
                        dates=chunk,
                        currency=asset_upper,
                        target_surfaces=target_surfaces,
                        model=model,
                        pn=pn,
                        yn=yn,
                        device=dev,
                        prior_batch=prior_chunk_tensor,
                        reg_weights=torch.tensor(reg_weights, dtype=torch.float32, device=dev),
                        tg=tg
                    )

                existing_results.extend(chunk_results)
                existing_results.sort(key=lambda r: r.date)

                # Save progress incrementally
                save_results(existing_results, str(file_path))
                
                # CUDA cache cleanup & GC to prevent VRAM fragmentation
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        study_results[asset_upper] = results_to_dataframe(existing_results)

    return study_results


def run_pettitt_test(H_series: np.ndarray) -> Tuple[int, float, float]:
    """
    Pure Python Pettitt test for change-point detection in the Hurst series.
    Returns: change_point_index, p_value, statistic
    """
    H_series = H_series[~np.isnan(H_series)]
    n = len(H_series)
    if n < 4:
        return 0, 1.0, 0.0

    diff = H_series[:, None] - H_series[None, :]
    D = np.sign(diff)
    V = D.sum(axis=1)
    U = np.cumsum(V)[:-1]

    abs_U = np.abs(U)
    K = np.max(abs_U)
    tau = int(np.argmax(abs_U)) + 1

    p_value = 2.0 * np.exp(-6.0 * (K ** 2) / (n ** 3 + n ** 2))
    p_value = min(p_value, 1.0)
    return tau, float(p_value), float(K)


def compute_hurst_statistics(df: pd.DataFrame) -> dict:
    """
    Computes statistical metrics for Hurst exponent from a calibration results DataFrame.
    Expects df to contain an 'H' column.

    Returns a dict with: mean, std, and autocorrelation coefficients at lags 1, 5, 10, 20.
    """
    if df.empty or "H" not in df.columns:
        return {
            "mean": 0.0,
            "std": 0.0,
            "autocorr_lag1": 0.0,
            "autocorr_lag5": 0.0,
            "autocorr_lag10": 0.0,
            "autocorr_lag20": 0.0,
        }

    H_series = df["H"].dropna()
    if len(H_series) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "autocorr_lag1": 0.0,
            "autocorr_lag5": 0.0,
            "autocorr_lag10": 0.0,
            "autocorr_lag20": 0.0,
        }

    mean_val = float(H_series.mean())
    std_val = float(H_series.std())

    # Calculate autocorr at lags 1, 5, 10, 20
    autocorr = {}
    for lag in [1, 5, 10, 20]:
        if len(H_series) > lag:
            val = H_series.autocorr(lag=lag)
            autocorr[f"autocorr_lag{lag}"] = float(val) if np.isfinite(val) else 0.0
        else:
            autocorr[f"autocorr_lag{lag}"] = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        **autocorr
    }


def plot_hurst_series(study_results: Dict[str, pd.DataFrame], save_path: Optional[str] = None) -> None:
    """
    Plots the calibrated Hurst exponent series for all assets in a single plot.
    """
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for asset, df in study_results.items():
        if df.empty or "H" not in df.columns:
            continue
        # Ensure df is sorted by date
        df_sorted = df.sort_values("date")
        # Parse dates for nice plotting
        dates = pd.to_datetime(df_sorted["date"])
        ax.plot(dates, df_sorted["H"], label=asset, alpha=0.8, linewidth=1.5)

    ax.set_title("Calibrated Hurst Exponent (H) Series (2020-2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hurst Exponent (H)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Hurst series plot to {save_path}")
    plt.close(fig)


def plot_hurst_correlation(study_results: Dict[str, pd.DataFrame], save_path: Optional[str] = None) -> None:
    """
    Computes correlation matrix of calibrated Hurst exponents and plots a correlation heatmap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Align the H series across assets into a single DataFrame
    h_data = {}
    for asset, df in study_results.items():
        if df.empty or "H" not in df.columns:
            continue
        h_data[asset] = df.set_index("date")["H"]

    if not h_data:
        warnings.warn("No data available for correlation plotting")
        return

    combined_df = pd.DataFrame(h_data).sort_index()
    corr_matrix = combined_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)

    # Tick labels
    ticks = np.arange(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)

    # Annotate correlation values in the cells
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black")

    ax.set_title("Cross-Asset Implied Hurst Exponent Correlation Heatmap")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation heatmap to {save_path}")
    plt.close(fig)
