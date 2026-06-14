"""
calibrate_fast.py — Newton-Raphson calibration using FNO autograd Jacobians.

Differential Machine Learning approach (Huge & Savine, 2020):
Instead of finite-difference gradients (6 extra forward passes), compute the
exact Jacobian ∂IV_FNO/∂θ via torch.autograd in a single backward pass through
the trained FNO surrogate.

Benefits over L-BFGS:
  - Jacobian cost: 1 backward pass vs 10 forward passes for 5-param 5-pt FD
  - Convergence: Newton step has quadratic convergence near the solution
  - Noise-free: analytical derivatives through smooth FNO (no FD discretization)

Usage:
    from calibrate_fast import calibrate_newton, benchmark_jacobian_speed
"""

import os
import sys
import time

import numpy as np
import torch
from torch.func import jacfwd
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrate import (
    _load_normalizers,
    _make_spatial_input,
    _fno_predict_real_iv,
    _BOUNDS_LOWER,
    _BOUNDS_UPPER,
)

# Ghost parameter values (fixed in 3D reparameterized calibration)
_GHOST_KAPPA = 1.0
_GHOST_THETA = 0.08
_GHOST_H     = 0.08

# 3D calibration bounds: [v0, zeta=sigma*rho, lam=sigma*sqrt(1-rho^2)]
_BOUNDS_LOWER_3D = torch.tensor([0.01, -0.90, 0.01])
_BOUNDS_UPPER_3D = torch.tensor([0.15, -0.01, 0.99])


def _reparam_to_6d(v0: torch.Tensor, zeta: torch.Tensor,
                   lam: torch.Tensor, device) -> torch.Tensor:
    """Back-transform (v₀, ζ, λ) → raw 6D parameter vector (B,6)."""
    sigma = torch.sqrt(zeta**2 + lam**2).clamp(min=0.01)
    rho   = (zeta / sigma).clamp(-0.9, -0.1)
    kappa = torch.full_like(v0, _GHOST_KAPPA)
    theta = torch.full_like(v0, _GHOST_THETA)
    H     = torch.full_like(v0, _GHOST_H)
    return torch.stack([kappa, theta, sigma, rho, v0, H], dim=-1).to(device)

__all__ = [
    "calibrate_newton",
    "fno_jacobian_autograd",
    "benchmark_jacobian_speed",
]

# ---------------------------------------------------------------------------
# Autograd Jacobian
# ---------------------------------------------------------------------------

def fno_jacobian_autograd(model, params_3d: torch.Tensor,
                           spatial: torch.Tensor) -> torch.Tensor:
    """
    Compute ∂IV_FNO/∂(v0, zeta, lam) analytically via forward-mode AD.

    Uses torch.func.jacfwd which dispatches n_inputs=3 JVPs (forward passes
    with tangent propagation). This is optimal for n_inputs << n_outputs:
      - jacfwd: 3 JVPs   ≈ 3× cost_forward
      - jacrev: 88 VJPs  ≈ 88× cost_backward  (old approach, 20-30× slower)

    Parameters
    ----------
    model      : loaded FiLM-FNO model (eval mode)
    params_3d  : (3,) tensor [v0, zeta, lam] (detached, clipped to bounds)
    spatial    : (1, nT, nK, 2) spatial coordinate tensor on same device

    Returns
    -------
    J : (nT, nK, 3) Jacobian — J[t,k,j] = ∂IV[t,k]/∂params_3d[j]
    """
    _load_normalizers()
    device = next(model.parameters()).device

    lo = _BOUNDS_LOWER_3D.to(device)
    hi = _BOUNDS_UPPER_3D.to(device)

    def _iv_flat(p3: torch.Tensor) -> torch.Tensor:
        """(3,) → (nT*nK,)  real IV, fully differentiable."""
        v0   = p3[0:1].clamp(lo[0], hi[0])
        zeta = p3[1:2].clamp(lo[1], hi[1])
        lam  = p3[2:3].clamp(lo[2], hi[2])
        p6   = _reparam_to_6d(v0, zeta, lam, device)
        iv   = _fno_predict_real_iv(model, p6, spatial.to(device))
        return iv.reshape(-1)

    # jacfwd: 3 forward-mode JVPs (one per input dimension)
    # No requires_grad needed — forward mode propagates tangents, not gradients
    p3 = params_3d.to(device).float().detach()
    J  = jacfwd(_iv_flat)(p3)                  # (nT*nK, 3)

    nT = spatial.shape[1]   # 8 maturities
    nK = spatial.shape[2]   # 11 strikes
    return J.reshape(nT, nK, 3).detach()


# ---------------------------------------------------------------------------
# Newton-Raphson calibration
# ---------------------------------------------------------------------------

def calibrate_newton(model, target_iv: np.ndarray,
                     T_grid, K_grid,
                     max_iter: int = 20,
                     tol: float = 1e-5,
                     damping: float = 0.5,
                     verbose: bool = False) -> dict:
    """
    Gauss-Newton calibration using autograd Jacobians through the FNO.

    Solves: min_{v0,zeta,lam} ||IV_FNO(v0,zeta,lam) - IV_target||²_F

    Algorithm (damped Gauss-Newton with Levenberg-Marquardt regularization):
        δθ = -(JᵀJ + εI)⁻¹ Jᵀ r      (Gauss-Newton normal equations)
        α  = line-search step size     (backtracking, starting from `damping`)
        θ  ← clip(θ + α·δθ, bounds)

    Parameters
    ----------
    model      : FiLM-FNO model (eval mode, loaded)
    target_iv  : (nT, nK) ndarray of market implied vols
    T_grid, K_grid : maturity and log-moneyness grids
    max_iter   : maximum Gauss-Newton iterations
    tol        : convergence tolerance on MSE
    damping    : initial step size ∈ (0,1]

    Returns
    -------
    dict: v0, zeta, lambda, sigma, rho, history, elapsed, n_iter, final_mse
    """
    model.eval()
    _load_normalizers()
    device   = next(model.parameters()).device
    spatial  = _make_spatial_input(T_grid, K_grid, device)
    target_t = torch.tensor(target_iv, dtype=torch.float32, device=device)

    T_arr   = np.asarray(T_grid)
    K_arr   = np.asarray(K_grid)
    atm_idx = int(np.argmin(np.abs(K_arr)))
    t01_idx = int(np.argmin(np.abs(T_arr - 0.1)))
    iv_short = float(target_iv[t01_idx, atm_idx])

    lo = _BOUNDS_LOWER_3D.numpy()
    hi = _BOUNDS_UPPER_3D.numpy()

    # 3 diverse starting points
    v0_est = float(np.clip(iv_short**2, 0.01, 0.14))
    inits  = np.array([
        [v0_est, -0.25, 0.35],
        [v0_est, -0.15, 0.50],
        [v0_est, -0.40, 0.25],
    ], dtype=np.float32)
    inits = np.clip(inits, lo + 1e-4, hi - 1e-4)

    best_loss   = float("inf")
    best_params = inits[0].copy()
    best_hist   = []
    best_n      = 0
    start_t     = time.time()

    for init in inits:
        theta = init.copy()
        hist  = []
        n     = 0

        for it in range(max_iter):
            n = it + 1
            theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
            lo_t = _BOUNDS_LOWER_3D.to(device)
            hi_t = _BOUNDS_UPPER_3D.to(device)
            theta_c = theta_t.clamp(lo_t, hi_t)

            with torch.no_grad():
                # theta_c slices are (1,) — correct input to _reparam_to_6d
                p6      = _reparam_to_6d(theta_c[0:1], theta_c[1:2],
                                         theta_c[2:3], device)
                iv_pred = _fno_predict_real_iv(model, p6, spatial)

            r    = (iv_pred - target_t).reshape(-1)
            loss = float((r**2).mean())
            hist.append(loss)

            if verbose:
                print(f"  [{it:2d}] loss={loss:.2e}  "
                      f"θ=[{theta_c[0]:.4f},{theta_c[1]:.4f},{theta_c[2]:.4f}]")
            if loss < tol:
                break

            # Autograd Jacobian
            J      = fno_jacobian_autograd(model, theta_c.detach(), spatial)
            J_np   = J.reshape(-1, 3).cpu().numpy()   # (nT*nK, 3)
            r_np   = r.detach().cpu().numpy()

            # Levenberg-Marquardt: (JᵀJ + ε·diag(JᵀJ)) δ = -Jᵀr
            JtJ    = J_np.T @ J_np
            eps_lm = 1e-4 * np.diag(JtJ).mean()
            delta  = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_np.T @ r_np)

            # Backtracking line search
            alpha = damping
            for _ in range(8):
                theta_new = np.clip(theta_c.cpu().numpy() + alpha * delta,
                                    lo + 1e-5, hi - 1e-5)
                tt = torch.tensor(theta_new, dtype=torch.float32, device=device)
                with torch.no_grad():
                    p6n  = _reparam_to_6d(tt[0:1], tt[1:2], tt[2:3], device)
                    ivn  = _fno_predict_real_iv(model, p6n, spatial)
                    ln   = float(((ivn - target_t)**2).mean())
                if ln < loss:
                    theta = theta_new
                    break
                alpha *= 0.5
            else:
                theta = theta_c.cpu().numpy()   # no improvement — keep current

        if hist and hist[-1] < best_loss:
            best_loss   = hist[-1]
            best_params = theta.copy()
            best_hist   = hist
            best_n      = n

    elapsed = time.time() - start_t
    v0_f, z_f, lm_f = best_params
    sigma_f = max(float(np.sqrt(z_f**2 + lm_f**2)), 0.01)
    rho_f   = float(np.clip(z_f / sigma_f, -0.9, -0.1))

    return {
        "v0":       float(v0_f),
        "zeta":     float(z_f),
        "lambda":   float(lm_f),
        "sigma":    sigma_f,
        "rho":      rho_f,
        "history":  best_hist,
        "elapsed":  elapsed,
        "n_iter":   best_n,
        "final_mse": best_loss,
    }


# ---------------------------------------------------------------------------
# Speed benchmark: autograd vs FD
# ---------------------------------------------------------------------------

def benchmark_jacobian_speed(model, T_grid, K_grid,
                              n_trials: int = 20) -> dict:
    """
    Wall-clock comparison: autograd Jacobian vs 5-point finite differences.

    Expected result: autograd ≈ 1 backward pass;
                     FD        ≈ 10 forward passes → 5-10× slower.
    """
    _load_normalizers()
    device  = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)
    rng     = np.random.default_rng(42)

    lo = _BOUNDS_LOWER_3D.numpy()
    hi = _BOUNDS_UPPER_3D.numpy()

    t_autograd, t_fd = [], []

    # Warmup: one call so jacfwd JIT-compiles the functional transform.
    # Without this, first-trial overhead (~100ms) skews the mean.
    _wp3 = torch.tensor(rng.uniform(lo + 0.01, hi - 0.01, 3).astype(np.float32),
                        device=device)
    fno_jacobian_autograd(model, _wp3, spatial)
    if str(device) == 'cuda': torch.cuda.synchronize()

    for _ in range(n_trials):
        p3 = torch.tensor(
            rng.uniform(lo + 0.01, hi - 0.01, 3).astype(np.float32),
            device=device,
        )

        # jacfwd timing (3 JVPs, one per input dim)
        if str(device) == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        _  = fno_jacobian_autograd(model, p3, spatial)
        if str(device) == 'cuda': torch.cuda.synchronize()
        t_autograd.append(time.perf_counter() - t0)

        # 5-point FD timing (4×3=12 forward passes)
        eps = np.array([5e-4, 5e-4, 5e-4])
        if str(device) == 'cuda': torch.cuda.synchronize()
        t0  = time.perf_counter()
        for j in range(3):
            for delta in (-2, -1, 1, 2):
                pp = p3.clone()
                pp[j] = float(np.clip(pp[j].item() + delta * eps[j],
                                      lo[j] + 1e-5, hi[j] - 1e-5))
                p6 = _reparam_to_6d(pp[0:1], pp[1:2], pp[2:3], device)
                with torch.no_grad():
                    _fno_predict_real_iv(model, p6, spatial)
        if str(device) == 'cuda': torch.cuda.synchronize()
        t_fd.append(time.perf_counter() - t0)

    speedup = float(np.mean(t_fd) / np.mean(t_autograd))
    print("=" * 56)
    print(" Jacobian Speed: jacfwd vs 5-point FD (3 params)")
    print("=" * 56)
    print(f"  Trials           : {n_trials}")
    print(f"  jacfwd mean      : {np.mean(t_autograd)*1e3:.2f} ms  (3 JVPs)")
    print(f"  FD (5-pt) mean   : {np.mean(t_fd)*1e3:.2f} ms  (12 fwd passes)")
    print(f"  Speedup          : {speedup:.1f}×  ({'✓' if speedup > 1.0 else '?'})")
    print("=" * 56)
    return {"t_autograd_ms": float(np.mean(t_autograd)*1e3),
            "t_fd_ms":       float(np.mean(t_fd)*1e3),
            "speedup":       speedup}



# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fno_model import MirrorPaddedFNO2d

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_GRID = np.linspace(-0.5, 0.5, 11)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading FNO v1 model ...")
    model = MirrorPaddedFNO2d()
    model.load_state_dict(torch.load("artifacts/models/fno_best.pth",
                                     map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    _load_normalizers()

    spatial = _make_spatial_input(T_GRID, K_GRID, device)

    # Synthetic target from true params — generate on GPU, move to numpy for calibrate_newton
    p_true = torch.tensor([[0.06, -0.20, 0.40]], dtype=torch.float32, device=device)
    p6_true = _reparam_to_6d(p_true[:, 0:1], p_true[:, 1:2],
                              p_true[:, 2:3], device)
    with torch.no_grad():
        iv_target = _fno_predict_real_iv(model, p6_true, spatial).cpu().numpy()

    # Speed benchmark (device auto-detected from model.parameters())
    stats = benchmark_jacobian_speed(model, T_GRID, K_GRID, n_trials=10)

    # Newton calibration
    print("\nRunning Newton calibration ...")
    result = calibrate_newton(model, iv_target, T_GRID, K_GRID,
                              max_iter=15, verbose=True)
    print(f"\nResult:")
    print(f"  v0    : {result['v0']:.4f}  (true=0.0600)")
    print(f"  zeta  : {result['zeta']:.4f}  (true=-0.2000)")
    print(f"  lambda: {result['lambda']:.4f}  (true=0.4000)")
    print(f"  sigma : {result['sigma']:.4f}  (true ~0.4472)")
    print(f"  rho   : {result['rho']:.4f}  (true ~-0.4472)")
    print(f"  MSE   : {result['final_mse']:.2e}")
    print(f"  iters : {result['n_iter']}")
    print(f"  time  : {result['elapsed']:.3f}s")
