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
    from deepvol.calibration.calibrate_newton import calibrate_newton, benchmark_jacobian_speed
"""

import os
import sys
import time

import numpy as np
import torch
from torch.func import jacfwd
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepvol.calibration.calibrate_bfgs import (
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
    "calibrate_heston",
    "calibrate_sabr",
    "calibrate_ssvi",
    "calibrate_rbergomi",
    "compute_local_vol_surface",
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
    _load_normalizers("v2")  # Jacobian uses same v2 normalizers as calibrate_newton
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
    _load_normalizers("v2")  # calibrate_newton uses FNO v2 (3-param, fix_H)
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
    best_theta_hist = []
    best_n      = 0
    start_t     = time.time()

    for init in inits:
        theta = init.copy()
        hist  = []
        theta_hist = []
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
            theta_hist.append(theta_c.cpu().numpy().copy())

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
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            eps_lm = max(eps_lm, 1e-12)
            try:
                delta  = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_np.T @ r_np)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                delta  = -np.linalg.pinv(JtJ + eps_lm * np.eye(3)) @ (J_np.T @ r_np)

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
            best_loss       = hist[-1]
            best_params     = theta.copy()
            best_hist       = hist
            best_theta_hist = theta_hist
            best_n          = n

    elapsed = time.time() - start_t
    v0_f, z_f, lm_f = best_params
    sigma_f = max(float(np.sqrt(z_f**2 + lm_f**2)), 0.01)
    rho_f   = float(np.clip(z_f / sigma_f, -0.9, -0.1))

    # Final forward pass to get the model-predicted IV surface
    best_v0_t = torch.tensor([v0_f], dtype=torch.float32, device=device)
    best_ze_t = torch.tensor([z_f], dtype=torch.float32, device=device)
    best_lm_t = torch.tensor([lm_f], dtype=torch.float32, device=device)
    with torch.no_grad():
        p6_best = _reparam_to_6d(best_v0_t, best_ze_t, best_lm_t, device)
        iv_fitted_t = _fno_predict_real_iv(model, p6_best, spatial)
    iv_fitted = iv_fitted_t.cpu().numpy().reshape(target_iv.shape)

    return {
        "v0":          float(v0_f),
        "zeta":        float(z_f),
        "lambda":      float(lm_f),
        "sigma":       sigma_f,
        "rho":         rho_f,
        "history":     best_hist,
        "theta_history": best_theta_hist,   # list of (3,) arrays [v0, zeta, lam]
        "elapsed":     elapsed,
        "n_iter":      best_n,
        "final_mse":   best_loss,
        "iv_fitted":   iv_fitted,
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
    _load_normalizers("v2")  # benchmark uses v2 normalizers
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
    from deepvol.calibration import calibrate_bfgs as _cal_mod  # noqa: F401 (kept for type-checker)
    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_GRID = np.linspace(-0.5, 0.5, 11)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load FNO v2 (N=40, N_cos=128, R²=0.9991) ────────────────────────────
    print("Loading FNO v2 model (N=40, N_cos=128, R²=0.9991) ...")
    model = MirrorPaddedFNO2d()
    model.load_state_dict(torch.load("artifacts/weights/fno_v2_final_prod.pth",
                                     map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Patch calibrate.py module-level paths to v2 normalizers before loading
    _load_normalizers(version="v2")

    spatial = _make_spatial_input(T_GRID, K_GRID, device)

    # Synthetic target from true params
    p_true = torch.tensor([[0.06, -0.20, 0.40]], dtype=torch.float32, device=device)
    p6_true = _reparam_to_6d(p_true[:, 0:1], p_true[:, 1:2],
                              p_true[:, 2:3], device)
    with torch.no_grad():
        iv_target = _fno_predict_real_iv(model, p6_true, spatial).cpu().numpy()

    # Speed benchmark (device auto-detected from model.parameters())
    stats = benchmark_jacobian_speed(model, T_GRID, K_GRID, n_trials=10)

    # Newton calibration
    print("\nRunning Newton calibration (v2 model) ...")
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


# ===========================================================================
# §5.1 Extension — Newton Calibration for Learnable H (4D free params)
# ===========================================================================

# 4D calibration bounds: [v0, zeta, lam, H]
_BOUNDS_LOWER_4D = torch.tensor([0.01, -0.90, 0.01, 0.04])
_BOUNDS_UPPER_4D = torch.tensor([0.15, -0.01, 0.99, 0.15])


def _reparam_to_6d_with_H(v0: torch.Tensor, zeta: torch.Tensor,
                           lam: torch.Tensor, H: torch.Tensor,
                           device) -> torch.Tensor:
    """Back-transform (v₀, ζ, λ, H) → raw 6D parameter vector (B,6)."""
    sigma = torch.sqrt(zeta**2 + lam**2).clamp(min=0.01)
    rho   = (zeta / sigma).clamp(-0.9, -0.1)
    kappa = torch.full_like(v0, _GHOST_KAPPA)
    theta = torch.full_like(v0, _GHOST_THETA)
    H_clp = H.clamp(0.04, 0.15)
    return torch.stack([kappa, theta, sigma, rho, v0, H_clp], dim=-1).to(device)


def calibrate_newton_h(model, iv_target: np.ndarray,
                       T_grid: np.ndarray, K_grid: np.ndarray,
                       max_iter: int = 20, tol: float = 1e-6,
                       eps_lm: float = 1e-4, verbose: bool = False,
                       init_v0: float = 0.07, init_zeta: float = -0.30,
                       init_lam: float = 0.40, init_H: float = 0.08,
                       ) -> dict:
    """
    Newton-Raphson calibration optimising 4D reparameterised space
    (v₀, ζ=σρ, λ=σ√(1−ρ²), H) — requires FNO v3 (param_dim=6).

    The Hurst exponent H is calibrated simultaneously with the variance
    and correlation parameters. The Fisher information matrix is 4×4.

    Parameters
    ----------
    model      : MirrorPaddedFNO2d(param_dim=6) loaded with v3 weights
    iv_target  : (nT, nK) ndarray of market implied vols
    T_grid     : (nT,) maturity grid
    K_grid     : (nK,) log-moneyness grid
    max_iter   : maximum Gauss-Newton iterations
    tol        : convergence tolerance on ||residual||
    eps_lm     : Levenberg-Marquardt regularisation
    verbose    : print per-iteration progress
    init_*     : starting values in (v₀, ζ, λ, H) space

    Returns
    -------
    dict with keys: v0, sigma, rho, H, zeta, lambda, final_mse,
                    n_iter, theta_history, converged
    """
    _load_normalizers("v3")  # calibrate_newton_h requires FNO v3 (param_dim=6)
    device   = next(model.parameters()).device
    spatial  = _make_spatial_input(T_grid, K_grid, device=device)
    iv_obs   = torch.tensor(iv_target.ravel(), dtype=torch.float32, device=device)
    nT, nK   = len(T_grid), len(K_grid)

    # Initial guess in 4D space
    theta = torch.tensor([init_v0, init_zeta, init_lam, init_H],
                         dtype=torch.float32, device=device, requires_grad=False)
    lo4 = _BOUNDS_LOWER_4D.to(device)
    hi4 = _BOUNDS_UPPER_4D.to(device)
    theta = theta.clamp(lo4, hi4)

    theta_history = [theta.detach().cpu().numpy().copy()]

    def _fwd(t):
        """Forward pass: 4D → 6D → normalised → FNO → real IV (flat)."""
        v0, zeta, lam, H = t[0], t[1], t[2], t[3]
        p6 = _reparam_to_6d_with_H(v0.unsqueeze(0), zeta.unsqueeze(0),
                                    lam.unsqueeze(0), H.unsqueeze(0), device)
        return _fno_predict_real_iv(model, p6, spatial).reshape(-1)

    def _jacobian(t):
        """(n_obs, 4) Jacobian via forward-mode AD — 4 JVPs."""
        t_leaf = t.detach().requires_grad_(True)
        J = torch.func.jacfwd(_fwd)(t_leaf)   # (n_obs, 4)
        return J.detach()

    converged = False
    for it in range(max_iter):
        iv_pred = _fwd(theta)
        residual = iv_pred - iv_obs        # (n_obs,)
        mse      = float((residual**2).mean().detach())

        if verbose:
            v0, ze, la, H = theta.tolist()
            sigma = float(torch.sqrt(torch.tensor(ze**2 + la**2)).clamp(min=0.01))
            rho   = float(torch.clamp(torch.tensor(ze / sigma), -0.9, -0.1))
            print(f"  it={it:2d}  mse={mse:.3e}  "
                  f"v0={v0:.4f} σ={sigma:.3f} ρ={rho:.3f} H={H:.4f}")

        if mse < tol:
            converged = True
            break

        J = _jacobian(theta)               # (n_obs, 4)
        JtJ  = J.T @ J                     # (4, 4)
        Jtr  = J.T @ residual              # (4,)

        # Levenberg-Marquardt regularisation
        lm   = eps_lm * torch.diag(JtJ).clamp(min=1e-8)
        JtJr = JtJ + torch.diag(lm)
        try:
            delta = torch.linalg.solve(JtJr, -Jtr)
        except torch.linalg.LinAlgError:
            delta = -Jtr * eps_lm          # gradient step fallback

        theta = (theta + delta).clamp(lo4, hi4)
        theta_history.append(theta.detach().cpu().numpy().copy())

    # Final forward pass
    iv_final = _fwd(theta).detach().cpu().numpy().reshape(nT, nK)
    v0, ze, la, H = [float(x) for x in theta.tolist()]
    sigma = float(np.sqrt(ze**2 + la**2))
    sigma = max(sigma, 0.01)
    rho   = float(np.clip(ze / sigma, -0.9, -0.1))

    return {
        "v0":           v0,
        "zeta":         ze,
        "lambda":       la,
        "sigma":        sigma,
        "rho":          rho,
        "H":            float(np.clip(H, 0.04, 0.15)),
        "final_mse":    float(((iv_final - iv_target)**2).mean()),
        "n_iter":       it + 1,
        "converged":    converged,
        "theta_history": theta_history,
        "iv_fitted":    iv_final,
    }


# ---------------------------------------------------------------------------
# §5.2 Phase 4 Model Zoo Calibrators
# ---------------------------------------------------------------------------

# ── Heston bounds and starts ────────────────────────────────────────────────
_BOUNDS_LOWER_HESTON = torch.tensor([0.5,  0.01, 0.1,  -0.95, 0.01])
_BOUNDS_UPPER_HESTON = torch.tensor([10.0, 0.25, 2.0,  -0.01, 0.25])

HESTON_STARTS = np.array([
    [2.0,  0.04, 0.50, -0.70, 0.04],   # typical SPX (low vol)
    [1.0,  0.15, 1.00, -0.80, 0.10],   # high vol regime
    [5.0,  0.08, 0.80, -0.60, 0.06],   # fast mean-reversion
    [0.8,  0.05, 0.30, -0.90, 0.03],   # low vol-of-vol
    [3.0,  0.10, 1.50, -0.50, 0.08],   # high vol-of-vol
], dtype=np.float32)

def calibrate_heston(model, iv_target: np.ndarray,
                     T_grid, K_grid,
                     max_iter: int = 30,
                     n_starts: int = 5,
                     verbose: bool = False) -> dict:
    """
    Calibrate Classic Heston to observed IV surface via Gauss-Newton on FNO surrogate.
    Optimizes: kappa, theta, log(sigma), rho, log(v0)
    """
    model.eval()
    _load_normalizers("heston")
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)
    target_t = torch.tensor(iv_target, dtype=torch.float32, device=device)
    iv_obs = target_t.reshape(-1)
    
    lo = _BOUNDS_LOWER_HESTON.to(device)
    hi = _BOUNDS_UPPER_HESTON.to(device)
    
    starts = HESTON_STARTS[:n_starts]
    if len(starts) < n_starts:
        np.random.seed(42)
        extra_starts = []
        for _ in range(n_starts - len(starts)):
            kappa_s = np.random.uniform(0.5, 10.0)
            theta_s = np.random.uniform(0.01, 0.25)
            sigma_s = np.exp(np.random.uniform(np.log(0.1), np.log(2.0)))
            rho_s = np.random.uniform(-0.95, -0.01)
            v0_s = np.exp(np.random.uniform(np.log(0.01), np.log(0.25)))
            extra_starts.append([kappa_s, theta_s, sigma_s, rho_s, v0_s])
        starts = np.vstack([starts, np.array(extra_starts, dtype=np.float32)])
        
    best_loss = float("inf")
    best_params = None
    best_hist = []
    best_loss_hist = []
    best_n = 0
    start_t = time.time()
    
    for init_idx, start in enumerate(starts):
        kappa_0, theta_0, sigma_0, rho_0, v0_0 = start
        p = torch.tensor([
            kappa_0,
            theta_0,
            np.log(max(sigma_0, 0.1)),
            rho_0,
            np.log(max(v0_0, 0.01))
        ], dtype=torch.float32, device=device)
        
        hist = []
        loss_hist = []
        n = 0
        
        for it in range(max_iter):
            n = it + 1
            
            kappa = p[0]
            theta = p[1]
            sigma = torch.exp(p[2])
            rho = p[3]
            v0 = torch.exp(p[4])
            
            kappa = torch.clamp(kappa, lo[0], hi[0])
            theta = torch.clamp(theta, lo[1], hi[1])
            sigma = torch.clamp(sigma, lo[2], hi[2])
            rho = torch.clamp(rho, lo[3], hi[3])
            v0 = torch.clamp(v0, lo[4], hi[4])
            
            p = torch.stack([kappa, theta, torch.log(sigma), rho, torch.log(v0)])
            
            with torch.no_grad():
                raw_params = torch.stack([kappa, theta, sigma, rho, v0]).unsqueeze(0)
                iv_pred = _fno_predict_real_iv(model, raw_params, spatial)
                r_pred = (iv_pred - target_t).reshape(-1)
                feller_viol = F.relu(sigma**2 - 2.0 * kappa * theta)
                loss = float((r_pred**2).mean().item() + 10.0 * (feller_viol**2).item())
                
            hist.append(p.detach().cpu().numpy().copy())
            loss_hist.append(loss)
            
            if verbose:
                print(f"  Start {init_idx} [{it:2d}] loss={loss:.2e}  "
                      f"θ=[{kappa:.4f},{theta:.4f},{sigma:.4f},{rho:.4f},{v0:.4f}]")
                      
            if loss < 1e-6:
                break
                
            def _res_vec(p_t):
                kp = p_t[0]
                th = p_t[1]
                sg = torch.exp(p_t[2])
                rh = p_t[3]
                v0_v = torch.exp(p_t[4])
                
                kp = torch.clamp(kp, lo[0], hi[0])
                th = torch.clamp(th, lo[1], hi[1])
                sg = torch.clamp(sg, lo[2], hi[2])
                rh = torch.clamp(rh, lo[3], hi[3])
                v0_v = torch.clamp(v0_v, lo[4], hi[4])
                
                raw = torch.stack([kp, th, sg, rh, v0_v]).unsqueeze(0)
                iv = _fno_predict_real_iv(model, raw, spatial).reshape(-1)
                r = iv - iv_obs
                
                f_viol = F.relu(sg**2 - 2.0 * kp * th)
                return torch.cat([r, 10.0 * f_viol.unsqueeze(0)])
                
            J = jacfwd(_res_vec)(p.detach())
            J_np = J.detach().cpu().numpy()
            
            with torch.no_grad():
                r_np = _res_vec(p.detach()).detach().cpu().numpy()
                
            JtJ = J_np.T @ J_np
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            eps_lm = max(eps_lm, 1e-12)
            
            try:
                delta = -np.linalg.solve(JtJ + eps_lm * np.eye(5), J_np.T @ r_np)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(JtJ + eps_lm * np.eye(5)) @ (J_np.T @ r_np)
                
            alpha = 0.5
            for _ in range(8):
                p_new_np = p.detach().cpu().numpy() + alpha * delta
                kp_new = np.clip(p_new_np[0], lo[0].item(), hi[0].item())
                th_new = np.clip(p_new_np[1], lo[1].item(), hi[1].item())
                sg_new = np.clip(np.exp(p_new_np[2]), lo[2].item(), hi[2].item())
                rh_new = np.clip(p_new_np[3], lo[3].item(), hi[3].item())
                v0_new = np.clip(np.exp(p_new_np[4]), lo[4].item(), hi[4].item())
                
                with torch.no_grad():
                    raw_new = torch.tensor([[kp_new, th_new, sg_new, rh_new, v0_new]], dtype=torch.float32, device=device)
                    ivn = _fno_predict_real_iv(model, raw_new, spatial)
                    r_new = (ivn - target_t).reshape(-1)
                    f_viol_new = max(0.0, sg_new**2 - 2.0 * kp_new * th_new)
                    loss_new = float((r_new**2).mean().item() + 10.0 * (f_viol_new**2))
                    
                if loss_new < loss:
                    p = torch.tensor([
                        kp_new,
                        th_new,
                        np.log(sg_new),
                        rh_new,
                        np.log(v0_new)
                    ], dtype=torch.float32, device=device)
                    break
                alpha *= 0.5
                
        kappa_f = float(np.clip(p[0].item(), lo[0].item(), hi[0].item()))
        theta_f = float(np.clip(p[1].item(), lo[1].item(), hi[1].item()))
        sigma_f = float(np.clip(np.exp(p[2].item()), lo[2].item(), hi[2].item()))
        rho_f = float(np.clip(p[3].item(), lo[3].item(), hi[3].item()))
        v0_f = float(np.clip(np.exp(p[4].item()), lo[4].item(), hi[4].item()))
        
        final_mse = loss_hist[-1]
        
        if final_mse < best_loss:
            best_loss = final_mse
            best_params = [kappa_f, theta_f, sigma_f, rho_f, v0_f]
            best_hist = hist
            best_loss_hist = loss_hist
            best_n = n
            
    elapsed = time.time() - start_t
    kappa_f, theta_f, sigma_f, rho_f, v0_f = best_params
    
    with torch.no_grad():
        raw_best = torch.tensor([[kappa_f, theta_f, sigma_f, rho_f, v0_f]], dtype=torch.float32, device=device)
        iv_fitted_t = _fno_predict_real_iv(model, raw_best, spatial)
    iv_fitted = iv_fitted_t.cpu().numpy().reshape(iv_target.shape)
    
    rmse_bps = float(np.sqrt(best_loss) * 10000.0)
    
    return {
        "params": {
            "kappa": kappa_f,
            "theta": theta_f,
            "sigma": sigma_f,
            "rho": rho_f,
            "v0": v0_f,
        },
        "param_vector": np.array(best_params),
        "loss": float(best_loss),
        "final_mse": float(best_loss),
        "rmse_bps": rmse_bps,
        "converged": bool(rmse_bps < 100.0),
        "message": "Optimization completed successfully" if rmse_bps < 100.0 else "Optimization did not converge within tolerance",
        "n_iter": best_n,
        "elapsed_ms": float(elapsed * 1000.0),
        "theta_history": [np.array([np.clip(x[0], lo[0].item(), hi[0].item()),
                                    np.clip(x[1], lo[1].item(), hi[1].item()),
                                    np.clip(np.exp(x[2]), lo[2].item(), hi[2].item()),
                                    np.clip(x[3], lo[3].item(), hi[3].item()),
                                    np.clip(np.exp(x[4]), lo[4].item(), hi[4].item())]) for x in best_hist],
        "loss_history": best_loss_hist,
        "iv_fitted": iv_fitted,
    }


# ── SABR bounds and starts ──────────────────────────────────────────────────
_BOUNDS_LOWER_SABR = torch.tensor([0.005, -0.95, 0.05])
_BOUNDS_UPPER_SABR = torch.tensor([0.5, 0.3, 1.5])

SABR_STARTS = np.array([
    [0.05, -0.5, 0.4],
    [0.15, -0.7, 0.8],
    [0.30, -0.3, 0.2]
], dtype=np.float32)

def calibrate_sabr(model, iv_target: np.ndarray,
                   T_grid, K_grid,
                   max_iter: int = 20,
                   n_starts: int = 3,
                   verbose: bool = False) -> dict:
    """
    Calibrate SABR model (beta=0.0) to observed IV surface.
    Optimizes: log(alpha), rho, log(nu)
    """
    model.eval()
    _load_normalizers("sabr")
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)
    target_t = torch.tensor(iv_target, dtype=torch.float32, device=device)
    iv_obs = target_t.reshape(-1)
    
    lo = _BOUNDS_LOWER_SABR.to(device)
    hi = _BOUNDS_UPPER_SABR.to(device)
    
    starts = SABR_STARTS[:n_starts]
    if len(starts) < n_starts:
        np.random.seed(42)
        extra_starts = []
        for _ in range(n_starts - len(starts)):
            alpha_s = np.exp(np.random.uniform(np.log(0.005), np.log(0.5)))
            rho_s = np.random.uniform(-0.95, 0.3)
            nu_s = np.exp(np.random.uniform(np.log(0.05), np.log(1.5)))
            extra_starts.append([alpha_s, rho_s, nu_s])
        starts = np.vstack([starts, np.array(extra_starts, dtype=np.float32)])
        
    best_loss = float("inf")
    best_params = None
    best_hist = []
    best_loss_hist = []
    best_n = 0
    start_t = time.time()
    
    for init_idx, start in enumerate(starts):
        alpha_0, rho_0, nu_0 = start
        p = torch.tensor([
            np.log(max(alpha_0, 0.005)),
            rho_0,
            np.log(max(nu_0, 0.05))
        ], dtype=torch.float32, device=device)
        
        hist = []
        loss_hist = []
        n = 0
        
        for it in range(max_iter):
            n = it + 1
            
            alpha = torch.exp(p[0])
            rho = p[1]
            nu = torch.exp(p[2])
            
            alpha = torch.clamp(alpha, lo[0], hi[0])
            rho = torch.clamp(rho, lo[1], hi[1])
            nu = torch.clamp(nu, lo[2], hi[2])
            
            p = torch.stack([torch.log(alpha), rho, torch.log(nu)])
            
            with torch.no_grad():
                raw_params = torch.stack([alpha, rho, nu]).unsqueeze(0)
                iv_pred = _fno_predict_real_iv(model, raw_params, spatial)
                r_pred = (iv_pred - target_t).reshape(-1)
                loss = float((r_pred**2).mean().item())
                
            hist.append(p.detach().cpu().numpy().copy())
            loss_hist.append(loss)
            
            if verbose:
                print(f"  Start {init_idx} [{it:2d}] loss={loss:.2e}  "
                      f"θ=[{alpha:.4f},{rho:.4f},{nu:.4f}]")
                      
            if loss < 1e-6:
                break
                
            def _res_vec(p_t):
                a_v = torch.exp(p_t[0])
                r_v = p_t[1]
                n_v = torch.exp(p_t[2])
                
                a_v = torch.clamp(a_v, lo[0], hi[0])
                r_v = torch.clamp(r_v, lo[1], hi[1])
                n_v = torch.clamp(n_v, lo[2], hi[2])
                
                raw = torch.stack([a_v, r_v, n_v]).unsqueeze(0)
                return _fno_predict_real_iv(model, raw, spatial).reshape(-1) - iv_obs
                
            J = jacfwd(_res_vec)(p.detach())
            J_np = J.detach().cpu().numpy()
            
            with torch.no_grad():
                r_np = _res_vec(p.detach()).detach().cpu().numpy()
                
            JtJ = J_np.T @ J_np
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            eps_lm = max(eps_lm, 1e-12)
            
            try:
                delta = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_np.T @ r_np)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(JtJ + eps_lm * np.eye(3)) @ (J_np.T @ r_np)
                
            alpha_d = 0.5
            for _ in range(8):
                p_new_np = p.detach().cpu().numpy() + alpha_d * delta
                a_new = np.clip(np.exp(p_new_np[0]), lo[0].item(), hi[0].item())
                r_new = np.clip(p_new_np[1], lo[1].item(), hi[1].item())
                n_new = np.clip(np.exp(p_new_np[2]), lo[2].item(), hi[2].item())
                
                with torch.no_grad():
                    raw_new = torch.tensor([[a_new, r_new, n_new]], dtype=torch.float32, device=device)
                    ivn = _fno_predict_real_iv(model, raw_new, spatial)
                    loss_new = float(((ivn - target_t)**2).mean().item())
                    
                if loss_new < loss:
                    p = torch.tensor([
                        np.log(a_new),
                        r_new,
                        np.log(n_new)
                    ], dtype=torch.float32, device=device)
                    break
                alpha_d *= 0.5
                
        alpha_f = float(np.clip(np.exp(p[0].item()), lo[0].item(), hi[0].item()))
        rho_f = float(np.clip(p[1].item(), lo[1].item(), hi[1].item()))
        nu_f = float(np.clip(np.exp(p[2].item()), lo[2].item(), hi[2].item()))
        
        final_mse = loss_hist[-1]
        
        if final_mse < best_loss:
            best_loss = final_mse
            best_params = [alpha_f, rho_f, nu_f]
            best_hist = hist
            best_loss_hist = loss_hist
            best_n = n
            
    elapsed = time.time() - start_t
    alpha_f, rho_f, nu_f = best_params
    
    with torch.no_grad():
        raw_best = torch.tensor([[alpha_f, rho_f, nu_f]], dtype=torch.float32, device=device)
        iv_fitted_t = _fno_predict_real_iv(model, raw_best, spatial)
    iv_fitted = iv_fitted_t.cpu().numpy().reshape(iv_target.shape)
    
    rmse_bps = float(np.sqrt(best_loss) * 10000.0)
    
    return {
        "alpha": alpha_f,
        "rho": rho_f,
        "nu": nu_f,
        "final_mse": float(best_loss),
        "rmse_bps": rmse_bps,
        "converged": bool(rmse_bps < 100.0),
        "n_iter": best_n,
        "elapsed_ms": float(elapsed * 1000.0),
        "theta_history": [np.array([np.clip(np.exp(x[0]), lo[0].item(), hi[0].item()),
                                    np.clip(x[1], lo[1].item(), hi[1].item()),
                                    np.clip(np.exp(x[2]), lo[2].item(), hi[2].item())]) for x in best_hist],
        "loss_history": best_loss_hist,
        "iv_fitted": iv_fitted,
    }


# ── SSVI bounds and starts ──────────────────────────────────────────────────
_BOUNDS_LOWER_SSVI = torch.tensor([-0.9, 0.05, 0.1])
_BOUNDS_UPPER_SSVI = torch.tensor([0.9, 4.0, 0.5])

SSVI_STARTS = np.array([
    [-0.4, 0.5, 0.3],
    [-0.7, 0.8, 0.45],
    [-0.1, 0.3, 0.15],
    [-0.5, 1.2, 0.25],
    [-0.3, 0.6, 0.35]
], dtype=np.float32)

def calibrate_ssvi(model, iv_target: np.ndarray,
                   T_grid, K_grid,
                   theta_atm_init: np.ndarray = None,
                   max_iter: int = 20,
                   n_starts: int = 5,
                   verbose: bool = False) -> dict:
    """
    Calibrate SSVI to observed IV surface.
    theta_atm_init: optional pre-extracted ATM total variance per maturity.
                    If None, it is estimated from iv_target[:, K=0] * T_grid.
    Optimizes: rho, log(eta), gamma
    """
    model.eval()
    _load_normalizers("ssvi")
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)
    target_t = torch.tensor(iv_target, dtype=torch.float32, device=device)
    iv_obs = target_t.reshape(-1)
    
    if theta_atm_init is None:
        T_arr = np.asarray(T_grid)
        K_arr = np.asarray(K_grid)
        atm_idx = int(np.argmin(np.abs(K_arr)))
        iv_atm = iv_target[:, atm_idx]
        theta_atm_init = (iv_atm ** 2) * T_arr
        theta_atm_init = np.maximum.accumulate(theta_atm_init)
        theta_atm_init = np.clip(theta_atm_init, 1e-6, None)
        
    theta_atm_t = torch.tensor(theta_atm_init, dtype=torch.float32, device=device)
    
    lo = _BOUNDS_LOWER_SSVI.to(device)
    hi = _BOUNDS_UPPER_SSVI.to(device)
    
    starts = SSVI_STARTS[:n_starts]
    if len(starts) < n_starts:
        np.random.seed(42)
        extra_starts = []
        for _ in range(n_starts - len(starts)):
            rho_s = np.random.uniform(-0.9, 0.9)
            eta_s = np.exp(np.random.uniform(np.log(0.05), np.log(4.0)))
            gamma_s = np.random.uniform(0.1, 0.5)
            extra_starts.append([rho_s, eta_s, gamma_s])
        starts = np.vstack([starts, np.array(extra_starts, dtype=np.float32)])
        
    best_loss = float("inf")
    best_params = None
    best_hist = []
    best_loss_hist = []
    best_n = 0
    start_t = time.time()
    
    for init_idx, start in enumerate(starts):
        rho_0, eta_0, gamma_0 = start
        p = torch.tensor([
            rho_0,
            np.log(max(eta_0, 0.05)),
            gamma_0
        ], dtype=torch.float32, device=device)
        
        hist = []
        loss_hist = []
        n = 0
        
        for it in range(max_iter):
            n = it + 1
            
            rho = p[0]
            eta = torch.exp(p[1])
            gamma = p[2]
            
            rho = torch.clamp(rho, lo[0], hi[0])
            eta = torch.clamp(eta, lo[1], hi[1])
            gamma = torch.clamp(gamma, lo[2], hi[2])
            
            p = torch.stack([rho, torch.log(eta), gamma])
            
            with torch.no_grad():
                raw_params = torch.cat([theta_atm_t, torch.stack([rho, eta, gamma])])
                iv_pred = _fno_predict_real_iv(model, raw_params.unsqueeze(0), spatial)
                r_pred = (iv_pred - target_t).reshape(-1)
                arb_viol = F.relu(eta * (1.0 + torch.abs(rho)) - 2.0)
                loss = float((r_pred**2).mean().item() + 10.0 * (arb_viol**2).item())
                
            hist.append(p.detach().cpu().numpy().copy())
            loss_hist.append(loss)
            
            if verbose:
                print(f"  Start {init_idx} [{it:2d}] loss={loss:.2e}  "
                      f"θ=[{rho:.4f},{eta:.4f},{gamma:.4f}]")
                      
            if loss < 1e-6:
                break
                
            def _res_vec(p_t):
                rh = p_t[0]
                et = torch.exp(p_t[1])
                gm = p_t[2]
                
                rh = torch.clamp(rh, lo[0], hi[0])
                et = torch.clamp(et, lo[1], hi[1])
                gm = torch.clamp(gm, lo[2], hi[2])
                
                raw = torch.cat([theta_atm_t, torch.stack([rh, et, gm])]).unsqueeze(0)
                iv = _fno_predict_real_iv(model, raw, spatial).reshape(-1)
                r = iv - iv_obs
                
                arb_v = F.relu(et * (1.0 + torch.abs(rh)) - 2.0)
                return torch.cat([r, 10.0 * arb_v.unsqueeze(0)])
                
            J = jacfwd(_res_vec)(p.detach())
            J_np = J.detach().cpu().numpy()
            
            with torch.no_grad():
                r_np = _res_vec(p.detach()).detach().cpu().numpy()
                
            JtJ = J_np.T @ J_np
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            eps_lm = max(eps_lm, 1e-12)
            
            try:
                delta = -np.linalg.solve(JtJ + eps_lm * np.eye(3), J_np.T @ r_np)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(JtJ + eps_lm * np.eye(3)) @ (J_np.T @ r_np)
                
            alpha_d = 0.5
            for _ in range(8):
                p_new_np = p.detach().cpu().numpy() + alpha_d * delta
                rh_new = np.clip(p_new_np[0], lo[0].item(), hi[0].item())
                et_new = np.clip(np.exp(p_new_np[1]), lo[1].item(), hi[1].item())
                gm_new = np.clip(p_new_np[2], lo[2].item(), hi[2].item())
                
                with torch.no_grad():
                    raw_new = torch.cat([theta_atm_t, torch.tensor([rh_new, et_new, gm_new], dtype=torch.float32, device=device)]).unsqueeze(0)
                    ivn = _fno_predict_real_iv(model, raw_new, spatial)
                    r_new = (ivn - target_t).reshape(-1)
                    arb_viol_new = max(0.0, et_new * (1.0 + abs(rh_new)) - 2.0)
                    loss_new = float((r_new**2).mean().item() + 10.0 * (arb_viol_new**2))
                    
                if loss_new < loss:
                    p = torch.tensor([
                        rh_new,
                        np.log(et_new),
                        gm_new
                    ], dtype=torch.float32, device=device)
                    break
                alpha_d *= 0.5
                
        rho_f = float(np.clip(p[0].item(), lo[0].item(), hi[0].item()))
        eta_f = float(np.clip(np.exp(p[1].item()), lo[1].item(), hi[1].item()))
        gamma_f = float(np.clip(p[2].item(), lo[2].item(), hi[2].item()))
        
        final_mse = loss_hist[-1]
        
        if final_mse < best_loss:
            best_loss = final_mse
            best_params = [rho_f, eta_f, gamma_f]
            best_hist = hist
            best_loss_hist = loss_hist
            best_n = n
            
    elapsed = time.time() - start_t
    rho_f, eta_f, gamma_f = best_params
    
    with torch.no_grad():
        raw_best = torch.cat([theta_atm_t, torch.tensor([rho_f, eta_f, gamma_f], dtype=torch.float32, device=device)]).unsqueeze(0)
        iv_fitted_t = _fno_predict_real_iv(model, raw_best, spatial)
    iv_fitted = iv_fitted_t.cpu().numpy().reshape(iv_target.shape)
    
    rmse_bps = float(np.sqrt(best_loss) * 10000.0)
    
    return {
        "rho": rho_f,
        "eta": eta_f,
        "gamma": gamma_f,
        "theta_atm": theta_atm_init,
        "final_mse": float(best_loss),
        "rmse_bps": rmse_bps,
        "converged": bool(rmse_bps < 100.0),
        "n_iter": best_n,
        "elapsed_ms": float(elapsed * 1000.0),
        "theta_history": [np.array([np.clip(x[0], lo[0].item(), hi[0].item()),
                                    np.clip(np.exp(x[1]), lo[1].item(), hi[1].item()),
                                    np.clip(x[2], lo[2].item(), hi[2].item())]) for x in best_hist],
        "loss_history": best_loss_hist,
        "iv_fitted": iv_fitted,
    }


# ── Rough Bergomi bounds and starts ─────────────────────────────────────────
_BOUNDS_LOWER_RBERGOMI = torch.tensor([0.01, 0.04, 0.5, -0.95])
_BOUNDS_UPPER_RBERGOMI = torch.tensor([0.20, 0.15, 4.0, 0.0])

RBERGOMI_STARTS = np.array([
    [0.04, 0.07, 1.5, -0.7],
    [0.09, 0.10, 2.5, -0.6],
    [0.15, 0.05, 3.2, -0.85],
    [0.02, 0.12, 0.8, -0.5]
], dtype=np.float32)

def calibrate_rbergomi(model, iv_target: np.ndarray,
                       T_grid, K_grid,
                       max_iter: int = 20,
                       n_starts: int = 3,
                       verbose: bool = False) -> dict:
    """
    Calibrate Rough Bergomi to observed IV surface via Gauss-Newton on FNO surrogate.
    Optimizes: log(v0), H, log(eta), rho
    """
    model.eval()
    _load_normalizers("rbergomi")
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)
    target_t = torch.tensor(iv_target, dtype=torch.float32, device=device)
    iv_obs = target_t.reshape(-1)
    
    lo = _BOUNDS_LOWER_RBERGOMI.to(device)
    hi = _BOUNDS_UPPER_RBERGOMI.to(device)
    
    starts = RBERGOMI_STARTS[:n_starts]
    if len(starts) < n_starts:
        np.random.seed(42)
        extra_starts = []
        for _ in range(n_starts - len(starts)):
            v0_s = np.exp(np.random.uniform(np.log(0.01), np.log(0.20)))
            H_s = np.random.uniform(0.04, 0.15)
            eta_s = np.exp(np.random.uniform(np.log(0.5), np.log(4.0)))
            rho_s = np.random.uniform(-0.95, 0.0)
            extra_starts.append([v0_s, H_s, eta_s, rho_s])
        starts = np.vstack([starts, np.array(extra_starts, dtype=np.float32)])
        
    best_loss = float("inf")
    best_params = None
    best_hist = []
    best_loss_hist = []
    best_n = 0
    start_t = time.time()
    
    for init_idx, start in enumerate(starts):
        v0_0, H_0, eta_0, rho_0 = start
        p = torch.tensor([
            np.log(max(v0_0, 0.01)),
            H_0,
            np.log(max(eta_0, 0.5)),
            rho_0
        ], dtype=torch.float32, device=device)
        
        hist = []
        loss_hist = []
        n = 0
        
        for it in range(max_iter):
            n = it + 1
            
            v0 = torch.exp(p[0])
            H = p[1]
            eta = torch.exp(p[2])
            rho = p[3]
            
            v0 = torch.clamp(v0, lo[0], hi[0])
            H = torch.clamp(H, lo[1], hi[1])
            eta = torch.clamp(eta, lo[2], hi[2])
            rho = torch.clamp(rho, lo[3], hi[3])
            
            p = torch.stack([torch.log(v0), H, torch.log(eta), rho])
            
            with torch.no_grad():
                raw_params = torch.stack([v0, H, eta, rho]).unsqueeze(0)
                iv_pred = _fno_predict_real_iv(model, raw_params, spatial)
                r_pred = (iv_pred - target_t).reshape(-1)
                loss = float((r_pred**2).mean().item())
                
            hist.append(p.detach().cpu().numpy().copy())
            loss_hist.append(loss)
            
            if verbose:
                print(f"  Start {init_idx} [{it:2d}] loss={loss:.2e}  "
                      f"θ=[{v0:.4f},{H:.4f},{eta:.4f},{rho:.4f}]")
                      
            if loss < 1e-6:
                break
                
            def _res_vec(p_t):
                v_v = torch.exp(p_t[0])
                h_v = p_t[1]
                e_v = torch.exp(p_t[2])
                r_v = p_t[3]
                
                v_v = torch.clamp(v_v, lo[0], hi[0])
                h_v = torch.clamp(h_v, lo[1], hi[1])
                e_v = torch.clamp(e_v, lo[2], hi[2])
                r_v = torch.clamp(r_v, lo[3], hi[3])
                
                raw = torch.stack([v_v, h_v, e_v, r_v]).unsqueeze(0)
                return _fno_predict_real_iv(model, raw, spatial).reshape(-1) - iv_obs
                
            J = jacfwd(_res_vec)(p.detach())
            J_np = J.detach().cpu().numpy()
            
            with torch.no_grad():
                r_np = _res_vec(p.detach()).detach().cpu().numpy()
                
            JtJ = J_np.T @ J_np
            eps_lm = 1e-4 * np.diag(JtJ).mean() if JtJ.size > 0 else 1e-4
            eps_lm = max(eps_lm, 1e-12)
            
            try:
                delta = -np.linalg.solve(JtJ + eps_lm * np.eye(4), J_np.T @ r_np)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(JtJ + eps_lm * np.eye(4)) @ (J_np.T @ r_np)
                
            alpha_d = 0.5
            for _ in range(8):
                p_new_np = p.detach().cpu().numpy() + alpha_d * delta
                v_new = np.clip(np.exp(p_new_np[0]), lo[0].item(), hi[0].item())
                h_new = np.clip(p_new_np[1], lo[1].item(), hi[1].item())
                e_new = np.clip(np.exp(p_new_np[2]), lo[2].item(), hi[2].item())
                r_new = np.clip(p_new_np[3], lo[3].item(), hi[3].item())
                
                with torch.no_grad():
                    raw_new = torch.tensor([[v_new, h_new, e_new, r_new]], dtype=torch.float32, device=device)
                    ivn = _fno_predict_real_iv(model, raw_new, spatial)
                    loss_new = float(((ivn - target_t)**2).mean().item())
                    
                if loss_new < loss:
                    p = torch.tensor([
                        np.log(v_new),
                        h_new,
                        np.log(e_new),
                        r_new
                    ], dtype=torch.float32, device=device)
                    break
                alpha_d *= 0.5
                
        v0_f = float(np.clip(np.exp(p[0].item()), lo[0].item(), hi[0].item()))
        H_f = float(np.clip(p[1].item(), lo[1].item(), hi[1].item()))
        eta_f = float(np.clip(np.exp(p[2].item()), lo[2].item(), hi[2].item()))
        rho_f = float(np.clip(p[3].item(), lo[3].item(), hi[3].item()))
        
        final_mse = loss_hist[-1]
        
        if final_mse < best_loss:
            best_loss = final_mse
            best_params = [v0_f, H_f, eta_f, rho_f]
            best_hist = hist
            best_loss_hist = loss_hist
            best_n = n
            
    elapsed = time.time() - start_t
    v0_f, H_f, eta_f, rho_f = best_params
    
    with torch.no_grad():
        raw_best = torch.tensor([[v0_f, H_f, eta_f, rho_f]], dtype=torch.float32, device=device)
        iv_fitted_t = _fno_predict_real_iv(model, raw_best, spatial)
    iv_fitted = iv_fitted_t.cpu().numpy().reshape(iv_target.shape)
    
    rmse_bps = float(np.sqrt(best_loss) * 10000.0)
    
    return {
        "v0": v0_f,
        "H": H_f,
        "eta": eta_f,
        "rho": rho_f,
        "final_mse": float(best_loss),
        "rmse_bps": rmse_bps,
        "converged": bool(rmse_bps < 100.0),
        "n_iter": best_n,
        "elapsed_ms": float(elapsed * 1000.0),
        "theta_history": [np.array([np.clip(np.exp(x[0]), lo[0].item(), hi[0].item()),
                                    np.clip(x[1], lo[1].item(), hi[1].item()),
                                    np.clip(np.exp(x[2]), lo[2].item(), hi[2].item()),
                                    np.clip(x[3], lo[3].item(), hi[3].item())]) for x in best_hist],
        "loss_history": best_loss_hist,
        "iv_fitted": iv_fitted,
    }


# ── Local Volatility helper ─────────────────────────────────────────────────

def compute_local_vol_surface(svi_params, T_grid, K_grid, use_fno: bool = False, model = None):
    """
    Compute Local Volatility surface from SVI parameters.
    
    Parameters:
    -----------
    svi_params : np.ndarray or torch.Tensor
        SVI parameters, shape (8, 5) or (40,) or (B, 8, 5) or (B, 40)
    T_grid : np.ndarray
        Maturity grid
    K_grid : np.ndarray
        Log-moneyness grid
    use_fno : bool
        If True, use the FNO surrogate model to predict the local vol surface.
        If False, apply the Dupire formula analytically.
    model : MirrorPaddedFNO2d, optional
        Loaded FNO surrogate model for local volatility (required if use_fno=True)
    """
    if not use_fno:
        is_numpy = isinstance(svi_params, np.ndarray)
        if is_numpy:
            if svi_params.ndim == 1:
                svi_params = svi_params.reshape(8, 5)
            elif svi_params.ndim == 2 and svi_params.shape[1] == 40:
                svi_params = svi_params.reshape(-1, 8, 5)
        else:
            if svi_params.ndim == 1:
                svi_params = svi_params.reshape(8, 5)
            elif svi_params.ndim == 2 and svi_params.shape[1] == 40:
                svi_params = svi_params.reshape(-1, 8, 5)
        
        from deepvol.models.local_vol import svi_to_lv_surface
        return svi_to_lv_surface(T_grid, K_grid, svi_params)
    
    else:
        assert model is not None, "FNO model must be provided when use_fno=True"
        device = next(model.parameters()).device
        
        if isinstance(svi_params, np.ndarray):
            svi_params_t = torch.tensor(svi_params, dtype=torch.float32, device=device)
        else:
            svi_params_t = svi_params.to(device)
            
        is_batched = (svi_params_t.ndim == 3) or (svi_params_t.ndim == 2 and svi_params_t.shape[0] > 1 and svi_params_t.shape[1] != 40)
        if svi_params_t.ndim == 3:
            svi_params_t = svi_params_t.reshape(svi_params_t.shape[0], -1)
        elif svi_params_t.ndim == 2 and svi_params_t.shape[1] == 5:
            svi_params_t = svi_params_t.reshape(-1).unsqueeze(0)
        elif svi_params_t.ndim == 1:
            svi_params_t = svi_params_t.unsqueeze(0)
            
        _load_normalizers("localvol")
        spatial = _make_spatial_input(T_grid, K_grid, device)
        
        with torch.no_grad():
            lv_surf_t = _fno_predict_real_iv(model, svi_params_t, spatial)
            
        if not is_batched:
            lv_surf_t = lv_surf_t.squeeze(0)
            
        if isinstance(svi_params, np.ndarray):
            return lv_surf_t.cpu().numpy()
        return lv_surf_t

