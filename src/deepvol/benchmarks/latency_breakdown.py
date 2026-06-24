"""
latency_breakdown.py — Latency profiling script for Rough Heston FNO calibration.

Profiles:
  - Single-surface Newton calibration (calibrate_newton)
  - Single-surface learnable H Newton calibration (calibrate_newton_h)
  - Batched Newton calibration (calibrate_batch) with batch sizes: 1, 4, 16, 32, 64

Measures CPU and GPU (if CUDA is available) latency using proper timing protocols.
Saves summary results to a JSON file.
"""

from __future__ import annotations

import os
import sys
import time
import json
import gc
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

# Setup python path dynamically relative to REPO ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Set working directory to repo root so all relative paths in deepvol work correctly
os.chdir(str(REPO_ROOT))

from deepvol.calibration.calibrate_bfgs import (
    _load_normalizers,
    _make_spatial_input,
    _fno_predict_real_iv,
)
from deepvol.calibration.calibrate_newton import (
    calibrate_newton,
    calibrate_newton_h,
    fno_jacobian_autograd,
    _BOUNDS_LOWER_3D,
    _BOUNDS_UPPER_3D,
    _BOUNDS_LOWER_4D,
    _BOUNDS_UPPER_4D,
    _reparam_to_6d,
    _reparam_to_6d_with_H,
)
from deepvol.calibration.batch_calibration import (
    calibrate_batch,
    calibrate_newton_batch,
    _get_assets,
    _make_spatial,
)


class DisableGC:
    """Context manager to temporarily disable garbage collection."""
    def __enter__(self):
        self.enabled = gc.isenabled()
        gc.disable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            gc.enable()


def sync(device: torch.device):
    """Enforce host-device synchronization if using CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_system_info() -> Dict[str, Any]:
    """Gather CPU, GPU, and system details for logging."""
    cpu = "Unknown"
    try:
        if platform.system() == "Linux":
            out = subprocess.check_output("grep -m 1 'model name' /proc/cpuinfo", shell=True).decode()
            cpu = out.split(":")[1].strip()
        else:
            cpu = platform.processor()
    except Exception:
        cpu = platform.processor() or "Unknown"

    gpu = "N/A"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)

    return {
        "os": platform.system(),
        "python_version": platform.python_version(),
        "cpu": cpu,
        "gpu": gpu,
        "cuda_available": torch.cuda.is_available(),
    }


# ─── Grid Definitions ──────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
K_GRID = np.linspace(-0.5, 0.5, 11, dtype=np.float32)


# ─── Calibration Loaders ───────────────────────────────────────────────────────
def load_fno_v2(device: torch.device):
    """Load FNO v2 model weights and normalizers."""
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    
    weights_path = REPO_ROOT / "artifacts" / "weights" / "fno_v2_final_prod.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"FNO v2 weights not found at: {weights_path}")
        
    model = MirrorPaddedFNO2d()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    _load_normalizers("v2")
    return model


def load_fno_v3(device: torch.device):
    """Load FNO v3 model weights and normalizers."""
    model, pn, yn, _ = _get_assets(str(device))
    return model, pn, yn


# ─── Profiling Methods ─────────────────────────────────────────────────────────

def profile_newton(model, target_iv: np.ndarray, device: torch.device, 
                   n_trials: int, n_trials_breakdown: int, n_warmup: int = 10) -> Dict[str, Any]:
    """Profile calibrate_newton (full + breakdown)."""
    # 1. Full Calibration
    # Warmup
    for _ in range(n_warmup):
        _ = calibrate_newton(model, target_iv, T_GRID, K_GRID, max_iter=20, tol=1e-5, damping=0.5, verbose=False)
        
    full_times = []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with DisableGC():
        for _ in range(n_trials):
            sync(device)
            t0 = time.perf_counter()
            _ = calibrate_newton(model, target_iv, T_GRID, K_GRID, max_iter=20, tol=1e-5, damping=0.5, verbose=False)
            sync(device)
            full_times.append(time.perf_counter() - t0)
            
    full_ms = np.array(full_times) * 1000.0
    
    # 2. Step Breakdown
    _load_normalizers("v2")
    spatial = _make_spatial_input(T_GRID, K_GRID, device)
    target_t = torch.tensor(target_iv, dtype=torch.float32, device=device)
    
    # Representative parameters
    theta = np.array([0.07, -0.25, 0.35], dtype=np.float32)
    theta_c = torch.tensor(theta, dtype=torch.float32, device=device)
    lo = _BOUNDS_LOWER_3D.numpy()
    hi = _BOUNDS_UPPER_3D.numpy()
    damping = 0.5
    
    # Single-step warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            p6 = _reparam_to_6d(theta_c[0:1], theta_c[1:2], theta_c[2:3], device)
            iv_pred = _fno_predict_real_iv(model, p6, spatial)
            r = (iv_pred - target_t).reshape(-1)
            loss_val = (r**2).mean()
        J = fno_jacobian_autograd(model, theta_c, spatial)
        J_np = J.reshape(-1, 3).cpu().numpy()
        r_np = r.detach().cpu().numpy()
        JtJ = J_np.T @ J_np
        eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
        eps_lm = max(eps_lm, 1e-12)
        delta = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_np.T @ r_np)
        for _ in range(8):
            theta_new = np.clip(theta_c.cpu().numpy() + damping * delta, lo + 1e-5, hi - 1e-5)
            tt = torch.tensor(theta_new, dtype=torch.float32, device=device)
            with torch.no_grad():
                p6n = _reparam_to_6d(tt[0:1], tt[1:2], tt[2:3], device)
                ivn = _fno_predict_real_iv(model, p6n, spatial)
                ln = float(((ivn - target_t)**2).mean())
            if ln < loss_val:
                break
                
    t_fwd, t_jac, t_sol, t_upd = [], [], [], []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with DisableGC():
        for _ in range(n_trials_breakdown):
            # FNO Model Forward Pass
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                p6 = _reparam_to_6d(theta_c[0:1], theta_c[1:2], theta_c[2:3], device)
                iv_pred = _fno_predict_real_iv(model, p6, spatial)
                r = (iv_pred - target_t).reshape(-1)
                loss_val = (r**2).mean()
            sync(device)
            t_fwd.append(time.perf_counter() - t0)
            
            # Jacobian Computation
            sync(device)
            t0 = time.perf_counter()
            J = fno_jacobian_autograd(model, theta_c, spatial)
            sync(device)
            t_jac.append(time.perf_counter() - t0)
            
            # LM Solver Step
            sync(device)
            t0 = time.perf_counter()
            J_np = J.reshape(-1, 3).cpu().numpy()
            r_np = r.detach().cpu().numpy()
            JtJ = J_np.T @ J_np
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            eps_lm = max(eps_lm, 1e-12)
            delta = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_np.T @ r_np)
            sync(device)
            t_sol.append(time.perf_counter() - t0)
            
            # Backtracking & Update
            sync(device)
            t0 = time.perf_counter()
            alpha = damping
            for _ in range(8):
                theta_new = np.clip(theta_c.cpu().numpy() + alpha * delta, lo + 1e-5, hi - 1e-5)
                tt = torch.tensor(theta_new, dtype=torch.float32, device=device)
                with torch.no_grad():
                    p6n = _reparam_to_6d(tt[0:1], tt[1:2], tt[2:3], device)
                    ivn = _fno_predict_real_iv(model, p6n, spatial)
                    ln = float(((ivn - target_t)**2).mean())
                if ln < loss_val:
                    break
                alpha *= 0.5
            sync(device)
            t_upd.append(time.perf_counter() - t0)
            
    return {
        "full_calibration": {
            "mean_ms": float(np.mean(full_ms)),
            "std_ms": float(np.std(full_ms)),
            "min_ms": float(np.min(full_ms)),
            "max_ms": float(np.max(full_ms)),
            "median_ms": float(np.median(full_ms)),
        },
        "breakdown_ms": {
            "fno_forward": float(np.mean(t_fwd) * 1000.0),
            "jacobian": float(np.mean(t_jac) * 1000.0),
            "lm_solver": float(np.mean(t_sol) * 1000.0),
            "parameter_update": float(np.mean(t_upd) * 1000.0),
            "total_step": float((np.mean(t_fwd) + np.mean(t_jac) + np.mean(t_sol) + np.mean(t_upd)) * 1000.0)
        }
    }


def profile_newton_h(model, target_iv: np.ndarray, device: torch.device, 
                     n_trials: int, n_trials_breakdown: int, n_warmup: int = 10) -> Dict[str, Any]:
    """Profile calibrate_newton_h (full + breakdown)."""
    # 1. Full Calibration
    # Warmup
    for _ in range(n_warmup):
        _ = calibrate_newton_h(model, target_iv, T_GRID, K_GRID, max_iter=20, tol=1e-6, eps_lm=1e-4, verbose=False)
        
    full_times = []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with DisableGC():
        for _ in range(n_trials):
            sync(device)
            t0 = time.perf_counter()
            _ = calibrate_newton_h(model, target_iv, T_GRID, K_GRID, max_iter=20, tol=1e-6, eps_lm=1e-4, verbose=False)
            sync(device)
            full_times.append(time.perf_counter() - t0)
            
    full_ms = np.array(full_times) * 1000.0
    
    # 2. Step Breakdown
    _load_normalizers("v3")
    spatial = _make_spatial_input(T_GRID, K_GRID, device=device)
    iv_obs = torch.tensor(target_iv.ravel(), dtype=torch.float32, device=device)
    
    theta = torch.tensor([0.07, -0.30, 0.40, 0.08], dtype=torch.float32, device=device)
    lo4 = _BOUNDS_LOWER_4D.to(device)
    hi4 = _BOUNDS_UPPER_4D.to(device)
    theta = theta.clamp(lo4, hi4)
    eps_lm = 1e-4

    def _fwd(t):
        v0, zeta, lam, H = t[0], t[1], t[2], t[3]
        p6 = _reparam_to_6d_with_H(v0.unsqueeze(0), zeta.unsqueeze(0),
                                    lam.unsqueeze(0), H.unsqueeze(0), device)
        return _fno_predict_real_iv(model, p6, spatial).reshape(-1)

    def _jacobian(t):
        t_leaf = t.detach().requires_grad_(True)
        J = torch.func.jacfwd(_fwd)(t_leaf)
        return J.detach()

    # Single-step warmup
    for _ in range(n_warmup):
        iv_pred = _fwd(theta)
        residual = iv_pred - iv_obs
        _ = (residual**2).mean().detach()
        J = _jacobian(theta)
        JtJ  = J.T @ J
        Jtr  = J.T @ residual
        lm   = eps_lm * torch.diag(JtJ).clamp(min=1e-8)
        JtJr = JtJ + torch.diag(lm)
        delta = torch.linalg.solve(JtJr, -Jtr)
        _ = (theta + delta).clamp(lo4, hi4)

    t_fwd, t_jac, t_sol, t_upd = [], [], [], []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with DisableGC():
        for _ in range(n_trials_breakdown):
            # FNO Model Forward Pass
            sync(device)
            t0 = time.perf_counter()
            iv_pred = _fwd(theta)
            residual = iv_pred - iv_obs
            _ = (residual**2).mean()
            sync(device)
            t_fwd.append(time.perf_counter() - t0)
            
            # Jacobian Computation
            sync(device)
            t0 = time.perf_counter()
            J = _jacobian(theta)
            sync(device)
            t_jac.append(time.perf_counter() - t0)
            
            # LM Solver Step
            sync(device)
            t0 = time.perf_counter()
            JtJ  = J.T @ J
            Jtr  = J.T @ residual
            lm   = eps_lm * torch.diag(JtJ).clamp(min=1e-8)
            JtJr = JtJ + torch.diag(lm)
            delta = torch.linalg.solve(JtJr, -Jtr)
            sync(device)
            t_sol.append(time.perf_counter() - t0)
            
            # Parameter Update (no line search)
            sync(device)
            t0 = time.perf_counter()
            _ = (theta + delta).clamp(lo4, hi4)
            sync(device)
            t_upd.append(time.perf_counter() - t0)
            
    return {
        "full_calibration": {
            "mean_ms": float(np.mean(full_ms)),
            "std_ms": float(np.std(full_ms)),
            "min_ms": float(np.min(full_ms)),
            "max_ms": float(np.max(full_ms)),
            "median_ms": float(np.median(full_ms)),
        },
        "breakdown_ms": {
            "fno_forward": float(np.mean(t_fwd) * 1000.0),
            "jacobian": float(np.mean(t_jac) * 1000.0),
            "lm_solver": float(np.mean(t_sol) * 1000.0),
            "parameter_update": float(np.mean(t_upd) * 1000.0),
            "total_step": float((np.mean(t_fwd) + np.mean(t_jac) + np.mean(t_sol) + np.mean(t_upd)) * 1000.0)
        }
    }


def profile_batch(model, pn, yn, target_iv: np.ndarray, device: torch.device, 
                  batch_size: int, n_trials: int, n_trials_breakdown: int, n_warmup: int = 10) -> Dict[str, Any]:
    """Profile calibrate_batch for a given batch size (full + breakdown)."""
    dates = [f"2024-01-{i:02d}" for i in range(1, batch_size + 1)]
    surfaces = {d: target_iv for d in dates}
    
    # 1. Full Calibration
    # Warmup
    for _ in range(n_warmup):
        _ = calibrate_batch(dates, currency="SPX", device=str(device), target_surfaces=surfaces, verbose=False)
        
    full_times = []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with DisableGC():
        for _ in range(n_trials):
            sync(device)
            t0 = time.perf_counter()
            _ = calibrate_batch(dates, currency="SPX", device=str(device), target_surfaces=surfaces, verbose=False)
            sync(device)
            full_times.append(time.perf_counter() - t0)
            
    full_ms = np.array(full_times) * 1000.0
    
    # 2. Step Breakdown
    B = batch_size
    target_iv_batch = np.stack([target_iv] * B, axis=0)
    target_iv_tensor = torch.tensor(target_iv_batch, dtype=torch.float32, device=device)
    target_flat = target_iv_tensor.reshape(B, 88)
    
    spatial_single = _make_spatial(device).squeeze(0)
    
    lo_t = torch.tensor([0.1, 0.01, 0.1, -0.9, 0.01, 0.04], dtype=torch.float32, device=device)
    hi_t = torch.tensor([5.0, 0.15, 1.0, -0.1, 0.15, 0.15], dtype=torch.float32, device=device)
    
    pn_mean = torch.tensor(pn.mean, dtype=torch.float32, device=device)
    pn_std = torch.tensor(pn.std, dtype=torch.float32, device=device)
    yn_mean = torch.tensor(yn.mean, dtype=torch.float32, device=device)
    yn_std = torch.tensor(yn.std, dtype=torch.float32, device=device)

    def fwd_fn(theta_single, spatial_single):
        theta_norm = (theta_single.unsqueeze(0) - pn_mean) / pn_std
        theta_norm = theta_norm.clamp(min=-3.0, max=3.0)
        spatial_input = spatial_single.unsqueeze(0)
        pred = model(spatial_input, theta_norm)
        iv = pred * yn_std + yn_mean
        return iv.clamp(min=1e-4).reshape(-1)

    vmap_fwd = torch.vmap(fwd_fn, in_dims=(0, 0))
    vmap_jac = torch.vmap(torch.func.jacfwd(fwd_fn, argnums=0), in_dims=(0, 0))
    
    inits = torch.tensor([
        [1.0, 0.08, 0.5, -0.5, 0.08, 0.08],
        [1.0, 0.08, 0.3, -0.7, 0.04, 0.06],
        [1.0, 0.08, 0.7, -0.3, 0.12, 0.10],
        [3.0, 0.03, 0.8, -0.4, 0.04, 0.12],
    ], dtype=torch.float32, device=device)
    
    num_starts = len(inits)
    theta = inits.repeat_interleave(B, dim=0)       # (M=B*num_starts, 6)
    target_expanded = target_flat.repeat(num_starts, 1)  # (M, 88)
    M = B * num_starts
    spatial_batch = spatial_single.unsqueeze(0).repeat(M, 1, 1, 1)
    
    chunk_sz = 4 if device.type == "cpu" else 128
    eps_lm = 1e-4
    loss_best = torch.full((M,), float('inf'), device=device)

    # Single-step warmup
    for _ in range(n_warmup):
        # fwd
        preds = []
        for i in range(0, M, chunk_sz):
            theta_sub = theta[i:i+chunk_sz]
            spatial_sub = spatial_batch[i:i+chunk_sz]
            preds.append(vmap_fwd(theta_sub, spatial_sub).detach())
        pred_val = torch.cat(preds, dim=0)
        
        # jac
        jacs = []
        for i in range(0, M, chunk_sz):
            theta_sub = theta[i:i+chunk_sz]
            spatial_sub = spatial_batch[i:i+chunk_sz]
            jacs.append(vmap_jac(theta_sub, spatial_sub).detach())
        jac_val = torch.cat(jacs, dim=0)
        
        # solver
        r = pred_val - target_expanded
        JtJ = torch.bmm(jac_val.transpose(1, 2), jac_val)
        Jtr = torch.bmm(jac_val.transpose(1, 2), r.unsqueeze(-1))
        diag_JtJ = torch.diagonal(JtJ, dim1=1, dim2=2)
        eps = eps_lm * diag_JtJ.clamp(min=1e-8) + 1e-9
        JtJ_reg = JtJ + torch.diag_embed(eps)
        delta = torch.linalg.solve(JtJ_reg, -Jtr).squeeze(-1)
        
        # update
        theta_best = theta.clone()
        alpha = torch.ones(M, 1, device=device)
        for ls_step in range(4):
            theta_cand = (theta + alpha * delta).clamp(lo_t + 1e-5, hi_t - 1e-5)
            preds_cand = []
            for i in range(0, M, chunk_sz):
                theta_sub = theta_cand[i:i+chunk_sz]
                spatial_sub = spatial_batch[i:i+chunk_sz]
                with torch.no_grad():
                    preds_cand.append(vmap_fwd(theta_sub, spatial_sub).detach())
            pred_cand_val = torch.cat(preds_cand, dim=0)
            loss_cand = ((pred_cand_val - target_expanded) ** 2).mean(dim=1)
            better = loss_cand < loss_best
            theta_best = torch.where(better.unsqueeze(-1), theta_cand, theta_best)
            loss_best = torch.where(better, loss_cand, loss_best)
            alpha = alpha * 0.5

    t_fwd, t_jac, t_sol, t_upd = [], [], [], []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with DisableGC():
        for _ in range(n_trials_breakdown):
            # FNO Model Forward Pass
            sync(device)
            t0 = time.perf_counter()
            preds = []
            for i in range(0, M, chunk_sz):
                theta_sub = theta[i:i+chunk_sz]
                spatial_sub = spatial_batch[i:i+chunk_sz]
                preds.append(vmap_fwd(theta_sub, spatial_sub).detach())
            pred_val = torch.cat(preds, dim=0)
            sync(device)
            t_fwd.append(time.perf_counter() - t0)
            
            # Jacobian Computation
            sync(device)
            t0 = time.perf_counter()
            jacs = []
            for i in range(0, M, chunk_sz):
                theta_sub = theta[i:i+chunk_sz]
                spatial_sub = spatial_batch[i:i+chunk_sz]
                jacs.append(vmap_jac(theta_sub, spatial_sub).detach())
            jac_val = torch.cat(jacs, dim=0)
            sync(device)
            t_jac.append(time.perf_counter() - t0)
            
            # LM Solver Step
            sync(device)
            t0 = time.perf_counter()
            r = pred_val - target_expanded
            JtJ = torch.bmm(jac_val.transpose(1, 2), jac_val)
            Jtr = torch.bmm(jac_val.transpose(1, 2), r.unsqueeze(-1))
            diag_JtJ = torch.diagonal(JtJ, dim1=1, dim2=2)
            eps = eps_lm * diag_JtJ.clamp(min=1e-8) + 1e-9
            JtJ_reg = JtJ + torch.diag_embed(eps)
            delta = torch.linalg.solve(JtJ_reg, -Jtr).squeeze(-1)
            sync(device)
            t_sol.append(time.perf_counter() - t0)
            
            # Backtracking Line Search & Clamping
            sync(device)
            t0 = time.perf_counter()
            theta_best = theta.clone()
            alpha = torch.ones(M, 1, device=device)
            for ls_step in range(4):
                theta_cand = (theta + alpha * delta).clamp(lo_t + 1e-5, hi_t - 1e-5)
                preds_cand = []
                for i in range(0, M, chunk_sz):
                    theta_sub = theta_cand[i:i+chunk_sz]
                    spatial_sub = spatial_batch[i:i+chunk_sz]
                    with torch.no_grad():
                        preds_cand.append(vmap_fwd(theta_sub, spatial_sub).detach())
                pred_cand_val = torch.cat(preds_cand, dim=0)
                loss_cand = ((pred_cand_val - target_expanded) ** 2).mean(dim=1)
                better = loss_cand < loss_best
                theta_best = torch.where(better.unsqueeze(-1), theta_cand, theta_best)
                loss_best = torch.where(better, loss_cand, loss_best)
                alpha = alpha * 0.5
            _ = theta_best.detach()
            sync(device)
            t_upd.append(time.perf_counter() - t0)
            
    return {
        "full_calibration": {
            "mean_ms": float(np.mean(full_ms)),
            "std_ms": float(np.std(full_ms)),
            "min_ms": float(np.min(full_ms)),
            "max_ms": float(np.max(full_ms)),
            "median_ms": float(np.median(full_ms)),
        },
        "breakdown_ms": {
            "fno_forward": float(np.mean(t_fwd) * 1000.0),
            "jacobian": float(np.mean(t_jac) * 1000.0),
            "lm_solver": float(np.mean(t_sol) * 1000.0),
            "parameter_update": float(np.mean(t_upd) * 1000.0),
            "total_step": float((np.mean(t_fwd) + np.mean(t_jac) + np.mean(t_sol) + np.mean(t_upd)) * 1000.0)
        }
    }


def compute_speedups(cpu_res: Dict[str, Any], gpu_res: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to calculate GPU speedup ratio over CPU."""
    speedups = {}
    
    # Full calibration speedup
    cpu_full = cpu_res["full_calibration"]["mean_ms"]
    gpu_full = gpu_res["full_calibration"]["mean_ms"]
    speedups["full_calibration"] = float(cpu_full / gpu_full) if gpu_full > 0 else 0.0
    
    # Breakdown speedups
    for key in cpu_res["breakdown_ms"]:
        cpu_val = cpu_res["breakdown_ms"][key]
        gpu_val = gpu_res["breakdown_ms"][key]
        speedups[key] = float(cpu_val / gpu_val) if gpu_val > 0 else 0.0
        
    return speedups


def main():
    parser = argparse.ArgumentParser(description="Rough Heston FNO Calibration Latency Profiler")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials for full calibration timing")
    parser.add_argument("--breakdown-trials", type=int, default=100, help="Number of trials for step breakdown timing")
    parser.add_argument("--warmups", type=int, default=10, help="Number of warmup passes before starting timing")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32, 64], help="Batch sizes to profile for calibrate_batch")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                        help="Device to target (auto selects cuda if available)")
    parser.add_argument("--output", type=str, default="", 
                        help="Output JSON path (defaults to standard derivatives-w4 path)")
    args = parser.parse_args()

    # Determine target devices
    if args.device == "auto":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
    elif args.device == "cpu":
        devices = ["cpu"]
    else:
        if not torch.cuda.is_available():
            print("ERROR: CUDA requested but not available.")
            sys.exit(1)
        devices = ["cuda"]

    print("=" * 60)
    print("      Rough Heston FNO Calibration Latency Profiler")
    print("=" * 60)
    print(f"Full Calibration Trials : {args.trials}")
    print(f"Step Breakdown Trials  : {args.breakdown_trials}")
    print(f"Target Devices         : {devices}")
    print("-" * 60)

    # 1. Fetch system metadata
    sys_info = get_system_info()
    print("System Metadata:")
    print(f"  OS             : {sys_info['os']}")
    print(f"  Python         : {sys_info['python_version']}")
    print(f"  CPU            : {sys_info['cpu']}")
    print(f"  GPU            : {sys_info['gpu']}")
    print(f"  CUDA Available : {sys_info['cuda_available']}")
    print("-" * 60)

    # Generate synthetic target surfaces to isolate mathematical throughput
    # We will generate a target using FNO v3 and a target using FNO v2
    fno_v2 = load_fno_v2(torch.device("cpu"))
    fno_v3, pn, yn = load_fno_v3(torch.device("cpu"))
    
    spatial_v2 = _make_spatial_input(T_GRID, K_GRID, torch.device("cpu"))
    p_true_v2 = torch.tensor([[0.06, -0.20, 0.40]], dtype=torch.float32)
    p6_true_v2 = _reparam_to_6d(p_true_v2[:, 0:1], p_true_v2[:, 1:2], p_true_v2[:, 2:3], torch.device("cpu"))
    
    spatial_v3 = _make_spatial(torch.device("cpu"))
    p_true_v3 = torch.tensor([[1.0, 0.08, 0.5, -0.5, 0.08, 0.08]], dtype=torch.float32)
    
    with torch.no_grad():
        target_iv_v2 = _fno_predict_real_iv(fno_v2, p6_true_v2, spatial_v2).cpu().numpy()
        
        norm_v3 = pn.transform_tensor(p_true_v3)
        pred_v3 = fno_v3(spatial_v3, norm_v3)
        target_iv_v3 = yn.inverse_transform_tensor(pred_v3).squeeze(0).clamp(min=1e-4).cpu().numpy()

    # Clear memory of CPU loaders
    del fno_v2, fno_v3, spatial_v2, spatial_v3
    gc.collect()

    results = {}

    # Define methods to profile
    methods = [
        "calibrate_newton",
        "calibrate_newton_h",
    ]
    for b_sz in args.batch_sizes:
        methods.append(f"calibrate_batch_{b_sz}")

    for m in methods:
        results[m] = {}

    for dev_name in devices:
        device = torch.device(dev_name)
        print(f"\n>>> Benchmarking on device: {dev_name.upper()} <<<")
        
        # Load FNO models on the targeted device
        model_v2 = load_fno_v2(device)
        model_v3, pn_v3, yn_v3 = load_fno_v3(device)
        
        # 1. calibrate_newton
        print("Profiling calibrate_newton...")
        res_newton = profile_newton(model_v2, target_iv_v2, device, args.trials, args.breakdown_trials, args.warmups)
        results["calibrate_newton"][dev_name] = res_newton
        print(f"  Full Calibration : {res_newton['full_calibration']['mean_ms']:.2f} ms")
        print(f"  Step Breakdown   : Forward={res_newton['breakdown_ms']['fno_forward']:.2f}ms, "
              f"Jacobian={res_newton['breakdown_ms']['jacobian']:.2f}ms, "
              f"Solver={res_newton['breakdown_ms']['lm_solver']:.2f}ms, "
              f"Update={res_newton['breakdown_ms']['parameter_update']:.2f}ms")

        # 2. calibrate_newton_h
        print("Profiling calibrate_newton_h...")
        res_newton_h = profile_newton_h(model_v3, target_iv_v3, device, args.trials, args.breakdown_trials, args.warmups)
        results["calibrate_newton_h"][dev_name] = res_newton_h
        print(f"  Full Calibration : {res_newton_h['full_calibration']['mean_ms']:.2f} ms")
        print(f"  Step Breakdown   : Forward={res_newton_h['breakdown_ms']['fno_forward']:.2f}ms, "
              f"Jacobian={res_newton_h['breakdown_ms']['jacobian']:.2f}ms, "
              f"Solver={res_newton_h['breakdown_ms']['lm_solver']:.2f}ms, "
              f"Update={res_newton_h['breakdown_ms']['parameter_update']:.2f}ms")

        # 3. calibrate_batch
        for b_sz in args.batch_sizes:
            print(f"Profiling calibrate_batch (batch size={b_sz})...")
            res_batch = profile_batch(model_v3, pn_v3, yn_v3, target_iv_v3, device, b_sz, args.trials, args.breakdown_trials, args.warmups)
            results[f"calibrate_batch_{b_sz}"][dev_name] = res_batch
            print(f"  Full Calibration : {res_batch['full_calibration']['mean_ms']:.2f} ms total "
                  f"({res_batch['full_calibration']['mean_ms']/b_sz:.2f} ms per surface)")
            print(f"  Step Breakdown   : Forward={res_batch['breakdown_ms']['fno_forward']:.2f}ms, "
                  f"Jacobian={res_batch['breakdown_ms']['jacobian']:.2f}ms, "
                  f"Solver={res_batch['breakdown_ms']['lm_solver']:.2f}ms, "
                  f"Update={res_batch['breakdown_ms']['parameter_update']:.2f}ms")

        # Clean device memory
        del model_v2, model_v3
        if dev_name == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Calculate speedups if CUDA was timed
    if "cuda" in devices and "cpu" in devices:
        print("\n" + "=" * 60)
        print("      CUDA Speedups (CPU / GPU)")
        print("=" * 60)
        for m in results:
            cpu_res = results[m]["cpu"]
            gpu_res = results[m]["cuda"]
            speedups = compute_speedups(cpu_res, gpu_res)
            results[m]["cuda"]["speedups"] = speedups
            print(f"{m:22s} : Full={speedups['full_calibration']:.2f}x, Forward={speedups['fno_forward']:.2f}x, "
                  f"Jacobian={speedups['jacobian']:.2f}x, Solver={speedups['lm_solver']:.2f}x, "
                  f"Update={speedups['parameter_update']:.2f}x")

    # Construct final payload
    payload = {
        "metadata": sys_info,
        "results": results
    }

    # Save to file
    out_path = args.output
    if not out_path:
        # Save to both locations: derivatives-w4 and derivatives
        out_paths = [
            Path("/home/execorn/programming/derivatives-w4/results/latency_breakdown/results.json"),
            Path("/home/execorn/programming/derivatives/results/latency_breakdown/results.json")
        ]
    else:
        out_paths = [Path(out_path)]

    for path in out_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Results successfully saved to: {path}")
        except Exception as exc:
            print(f"Warning: Failed to save results to {path}: {exc}")


if __name__ == "__main__":
    main()
