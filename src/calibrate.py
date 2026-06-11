"""
calibrate.py — L-BFGS calibration using the FiLM-conditioned FNO surrogate.

Interface change from previous version:
  - model.forward(spatial, theta_norm) instead of model.forward(fno_input)
  - ParameterNormalizer and IVSurfaceNormalizer are applied here:
      * input θ → z-score → model → z-score IV → denormalise → real IV
  - The calibration loss is computed in REAL IV space against the market surface
  - Jacobian confidence scores are computed in real IV space (denormalised)

Normalizer files are loaded once at module import and cached globally.
"""

import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizers import ParameterNormalizer, IVSurfaceNormalizer

# ─── Parameter bounds (must match LHS training distribution) ─────────────────
_BOUNDS_LOWER = torch.tensor([0.1,  0.01, 0.1,  -0.9, 0.01, 0.02])
_BOUNDS_UPPER = torch.tensor([5.0,  0.15, 1.0,  -0.1, 0.15, 0.15])
_PARAM_NAMES  = ["kappa", "theta", "sigma", "rho", "v0", "H"]

# ─── Lazy-load normalizers ────────────────────────────────────────────────────
_PARAM_NORM_PATH = "artifacts/models/param_normalizer.npz"
_IV_NORM_PATH    = "artifacts/models/iv_normalizer.npz"
_param_norm: ParameterNormalizer | None = None
_iv_norm:    IVSurfaceNormalizer | None = None


def _load_normalizers():
    global _param_norm, _iv_norm
    if _param_norm is None:
        if os.path.exists(_PARAM_NORM_PATH):
            _param_norm = ParameterNormalizer.load(_PARAM_NORM_PATH)
        else:
            # Fallback: identity normalizer (for models trained without normalizers)
            _param_norm = _IdentityParamNorm()
    if _iv_norm is None:
        if os.path.exists(_IV_NORM_PATH):
            _iv_norm = IVSurfaceNormalizer.load(_IV_NORM_PATH)
        else:
            _iv_norm = _IdentityIVNorm()


class _IdentityParamNorm:
    """No-op normalizer — used when normalizer files are missing (legacy model)."""
    def transform_tensor(self, t): return t
    def inverse_transform_tensor(self, t): return t


class _IdentityIVNorm:
    """No-op normalizer — used when normalizer files are missing (legacy model)."""
    def inverse_transform_tensor(self, t): return t


# ─── Coordinate grid helpers ─────────────────────────────────────────────────

def _make_spatial_input(T_grid: np.ndarray, K_grid: np.ndarray,
                        device: torch.device) -> torch.Tensor:
    """
    Build the (1, nT, nK, 2) spatial input [T_norm, K_norm] for the FNO.
    T is normalised to mean=0, std=1 over the training grid.
    K is normalised to [-1, 1] (already in [-0.5, 0.5], divide by 0.5).
    """
    T_arr = np.array(T_grid, dtype=np.float32)
    K_arr = np.array(K_grid, dtype=np.float32)

    # Same normalisation used during training
    T_norm = (T_arr - T_arr.mean()) / (T_arr.std() + 1e-8)
    K_norm = K_arr / 0.5

    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing="ij")  # (nT, nK)
    coords = np.stack([T_mesh, K_mesh], axis=-1)                  # (nT, nK, 2)
    return torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)  # (1,nT,nK,2)


def _fno_predict_real_iv(model, params_raw: torch.Tensor,
                         spatial: torch.Tensor) -> torch.Tensor:
    """
    Run one FNO forward pass and return denormalised IV surface.

    Parameters
    ----------
    params_raw : (6,) or (B, 6) in original parameter space
    spatial    : (1, nT, nK, 2) coordinate grid

    Returns
    -------
    iv_real : (nT, nK) or (B, nT, nK) — real implied volatility ≥ 1e-4
    """
    _load_normalizers()
    device = spatial.device

    if params_raw.dim() == 1:
        params_raw = params_raw.unsqueeze(0)   # (1, 6)

    # Z-score normalise parameters
    params_norm = _param_norm.transform_tensor(params_raw.to(torch.float32))

    # FNO forward — output is in normalised z-score IV space
    pred_norm = model(spatial.expand(params_norm.size(0), -1, -1, -1),
                      params_norm)            # (B, nT, nK)

    # Denormalise to real IV space
    iv_real = _iv_norm.inverse_transform_tensor(pred_norm)
    iv_real = iv_real.clamp(min=1e-4)         # enforce positivity

    return iv_real.squeeze(0)                 # (nT, nK)


# ─── Confidence scores ────────────────────────────────────────────────────────

def compute_confidence_scores(model, calibrated_params: np.ndarray,
                              T_grid, K_grid) -> dict[str, float]:
    """
    Jacobian column Frobenius norms in REAL IV space:
        score_i = ||∂IV_real/∂θᵢ||_F   for i in {κ, θ, σ, ρ, v₀, H}

    Scores are normalised so the most sensitive parameter = 1.0.

    The Jacobian is computed in real parameter space — the chain rule
    propagates through the z-score transforms automatically via Autograd.
    """
    model.eval()
    _load_normalizers()
    device = next(model.parameters()).device

    spatial = _make_spatial_input(T_grid, K_grid, device)  # (1, nT, nK, 2)
    params_t = torch.tensor(calibrated_params, dtype=torch.float32,
                            requires_grad=True)

    def _iv_flat(p):
        """Real-space IV, flattened: (6,) → (nT*nK,)."""
        return _fno_predict_real_iv(model, p, spatial).reshape(-1)

    J = torch.autograd.functional.jacobian(
        _iv_flat, params_t, create_graph=False, vectorize=False,
    )  # (nT*nK, 6)

    col_norms  = J.pow(2).sum(dim=0).sqrt()          # (6,)
    max_norm   = col_norms.max().clamp(min=1e-12)
    scores_raw = (col_norms / max_norm).detach().numpy()

    return {name: float(s) for name, s in zip(_PARAM_NAMES, scores_raw)}


# ─── L-BFGS calibration ───────────────────────────────────────────────────────

def calibrate_parameters(model, target_iv_surface: np.ndarray,
                         init_params: np.ndarray, T_grid, K_grid,
                         max_iter: int = 100, lr: float = 0.1):
    """
    Calibrate Rough Heston parameters to a market IV surface using L-BFGS.

    Optimisation is in logit-space to enforce strict feasibility w.r.t.
    the training bounds [lower, upper]. The FNO is queried in normalised
    space; the calibration loss is MSE in real IV space so that it is
    directly interpretable in volatility points.

    Returns
    -------
    final_params : (6,) ndarray in original space
    history      : list of loss values
    elapsed      : wall-clock seconds
    """
    model.eval()
    _load_normalizers()
    device = next(model.parameters()).device

    spatial = _make_spatial_input(T_grid, K_grid, device)  # (1, nT, nK, 2)
    target_t = torch.tensor(target_iv_surface, dtype=torch.float32, device=device)

    # Logit-space parameterisation
    init_t      = torch.tensor(init_params, dtype=torch.float32)
    init_t      = torch.clamp(init_t, _BOUNDS_LOWER + 1e-6, _BOUNDS_UPPER - 1e-6)
    init_scaled = (init_t - _BOUNDS_LOWER) / (_BOUNDS_UPPER - _BOUNDS_LOWER)
    init_scaled = torch.clamp(init_scaled, 1e-4, 1.0 - 1e-4)
    logits = torch.log(init_scaled / (1.0 - init_scaled))
    logits.requires_grad_(True)

    optimizer = optim.LBFGS(
        [logits], lr=lr, max_iter=20,
        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=10,
    )

    history = []

    def closure():
        optimizer.zero_grad()
        params_scaled = torch.sigmoid(logits)
        params_raw    = _BOUNDS_LOWER + params_scaled * (_BOUNDS_UPPER - _BOUNDS_LOWER)
        params_raw    = params_raw.to(device)

        # FNO forward in real IV space (normalisation applied inside)
        pred_iv = _fno_predict_real_iv(model, params_raw, spatial)  # (nT, nK)
        loss    = torch.nn.functional.mse_loss(pred_iv, target_t)
        loss.backward()
        return loss

    start = time.time()
    for _ in range(max_iter // 20):
        loss = optimizer.step(closure)
        history.append(loss.item())
        if loss.item() < 1e-6:
            break
    elapsed = time.time() - start

    final_scaled = torch.sigmoid(logits.detach())
    final_params = _BOUNDS_LOWER + final_scaled * (_BOUNDS_UPPER - _BOUNDS_LOWER)
    return final_params.numpy(), history, elapsed


# ─── Reparameterized 3D calibration ──────────────────────────────────────────
# The Lifted Rough Heston space (κ,θ,σ,ρ,v₀,H) has FIM condition number ~10⁸.
# Three parameters are "ghost" on T∈[0.1,2.0]:
#   κ  — fractional kernel dampens its effect to zero for T<2y
#   H  — roughness signature lives in T<0.04 (invisible at T_min=0.1)
#   σ,ρ individually — only ζ=σρ (skew driver) and λ=σ√(1-ρ²) are observable
#
# Optimising over (v₀, ζ, λ) drops condition number to ~10³–10⁴.
# Back-transform: σ = √(ζ²+λ²), ρ = ζ/σ.

_GHOST_KAPPA = 1.0
_GHOST_THETA = 0.08
_GHOST_H     = 0.08

# 3D bounds: [v0, zeta, lambda]
_BOUNDS_LOWER_3D = torch.tensor([0.01, -0.90, 0.01])
_BOUNDS_UPPER_3D = torch.tensor([0.15, -0.01, 0.99])


def _reparam_to_6d(v0: torch.Tensor, zeta: torch.Tensor,
                   lam: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Back-transform (v₀, ζ, λ) → full 6D parameter vector (B,6)."""
    sigma = torch.sqrt(zeta**2 + lam**2).clamp(min=0.01)
    rho   = (zeta / sigma).clamp(-0.9, -0.1)
    kappa = torch.full_like(v0, _GHOST_KAPPA)
    theta = torch.full_like(v0, _GHOST_THETA)
    H     = torch.full_like(v0, _GHOST_H)
    return torch.stack([kappa, theta, sigma, rho, v0, H], dim=-1).to(device)


def calibrate_reparameterized(model, target_iv_surface: np.ndarray,
                               T_grid, K_grid,
                               max_iter: int = 100, lr: float = 0.1) -> dict:
    """
    Calibrate Lifted Heston to a market IV surface in the identifiable 3D subspace.

    Fixes ghost parameters κ=1.0, θ=0.08, H=0.08 and optimises (v₀, ζ=σρ, λ=σ√(1-ρ²))
    in logit space via L-BFGS.  Back-transform: σ=√(ζ²+λ²), ρ=ζ/σ.

    Returns
    -------
    dict with keys: v0, zeta, lambda, sigma, rho, history (list), elapsed (s)
    """
    model.eval()
    _load_normalizers()
    device = next(model.parameters()).device

    spatial  = _make_spatial_input(T_grid, K_grid, device)
    target_t = torch.tensor(target_iv_surface, dtype=torch.float32, device=device)

    # ── Data-driven initial guess for v0 ─────────────────────────────────────
    # ATM IV ≈ sqrt(v0) for short maturities under Rough Heston.
    # (NOT divided by T — the IV level directly approximates sqrt(v0))
    T_arr   = np.asarray(T_grid)
    K_arr   = np.asarray(K_grid)
    atm_idx = int(np.argmin(np.abs(K_arr)))
    t01_idx = int(np.argmin(np.abs(T_arr - 0.1)))
    t10_idx = int(np.argmin(np.abs(T_arr - 1.0)))

    iv_atm_short = float(target_iv_surface[t01_idx, atm_idx])
    # IV_ATM ≈ sqrt(v0)  →  v0 ≈ IV_ATM² (correct formula, no divide by T)
    v0_est = float(np.clip(iv_atm_short ** 2, 0.01, 0.14))

    # ATM numerical skew at T=1 → rough estimate of zeta direction
    if 0 < atm_idx < len(K_arr) - 1:
        dk      = float(K_arr[atm_idx + 1] - K_arr[atm_idx - 1])
        iv_skew = (float(target_iv_surface[t10_idx, atm_idx + 1])
                   - float(target_iv_surface[t10_idx, atm_idx - 1])) / (dk + 1e-9)
        # typical C ≈ 0.5 relating skew to zeta; clip to feasible
        zeta_dd = float(np.clip(iv_skew / 0.5, -0.89, -0.02))
    else:
        zeta_dd = -0.25

    lo_np = _BOUNDS_LOWER_3D.numpy()
    hi_np = _BOUNDS_UPPER_3D.numpy()

    # 3 diverse starts: data-driven + two bracketing extremes
    inits_raw = [
        (v0_est, zeta_dd,  0.35),    # data-driven
        (v0_est, -0.20,    0.50),    # moderate baseline
        (v0_est, -0.40,    0.25),    # high skew / low lam
    ]
    inits = [
        (
            float(np.clip(v,  lo_np[0] + 1e-4, hi_np[0] - 1e-4)),
            float(np.clip(z,  lo_np[1] + 1e-4, hi_np[1] - 1e-4)),
            float(np.clip(lm, lo_np[2] + 1e-4, hi_np[2] - 1e-4)),
        )
        for v, z, lm in inits_raw
    ]

    def _to_logit(x, lo, hi):
        x_c = max(min(float(x), hi - 1e-5), lo + 1e-5)
        s   = (x_c - lo) / (hi - lo)
        return float(np.log(s / (1.0 - s)))

    def _run_lbfgs(v0_i, z_i, lm_i, n_outer):
        """L-BFGS from (v0_i, z_i, lm_i).  n_outer outer steps × 20 LBFGS evals each."""
        logits = torch.tensor(
            [_to_logit(v0_i, lo_np[0], hi_np[0]),
             _to_logit(z_i,  lo_np[1], hi_np[1]),
             _to_logit(lm_i, lo_np[2], hi_np[2])],
            dtype=torch.float32, requires_grad=True,
        )
        opt  = optim.LBFGS([logits], lr=lr, max_iter=20,
                            tolerance_grad=1e-7, tolerance_change=1e-9, history_size=10)
        hist = []

        def closure():
            opt.zero_grad()
            s    = torch.sigmoid(logits)
            lo_t = _BOUNDS_LOWER_3D.to(device)
            hi_t = _BOUNDS_UPPER_3D.to(device)
            p6   = _reparam_to_6d(
                (lo_t[0] + s[0] * (hi_t[0] - lo_t[0])).unsqueeze(0),
                (lo_t[1] + s[1] * (hi_t[1] - lo_t[1])).unsqueeze(0),
                (lo_t[2] + s[2] * (hi_t[2] - lo_t[2])).unsqueeze(0),
                device,
            )
            loss = torch.nn.functional.mse_loss(
                _fno_predict_real_iv(model, p6, spatial), target_t)
            loss.backward()
            return loss

        for _ in range(n_outer):
            loss = opt.step(closure)
            hist.append(loss.item())
            if loss.item() < 1e-9:
                break
        return loss.item(), logits.detach(), hist

    # Phase 1 — 3 starts × 3 outer steps (60 LBFGS evals each)
    start_t    = time.time()
    best_loss  = float("inf")
    best_logits = None
    all_hist   = []

    for v0_i, z_i, lm_i in inits:
        loss_i, logits_i, hist_i = _run_lbfgs(v0_i, z_i, lm_i, n_outer=3)
        all_hist.extend(hist_i)
        if loss_i < best_loss:
            best_loss   = loss_i
            best_logits = logits_i.clone()

    # Phase 2 — refine winner with 2 more outer steps
    with torch.no_grad():
        s_  = torch.sigmoid(best_logits)
        v0_r  = (lo_np[0] + s_[0].item() * (hi_np[0] - lo_np[0]))
        z_r   = (lo_np[1] + s_[1].item() * (hi_np[1] - lo_np[1]))
        lm_r  = (lo_np[2] + s_[2].item() * (hi_np[2] - lo_np[2]))
    loss_f, best_logits, hist_f = _run_lbfgs(v0_r, z_r, lm_r, n_outer=2)
    all_hist.extend(hist_f)

    elapsed = time.time() - start_t

    # Extract final parameters
    with torch.no_grad():
        s_    = torch.sigmoid(best_logits)
        lo_t  = _BOUNDS_LOWER_3D
        hi_t  = _BOUNDS_UPPER_3D
        v0_f  = (lo_t[0] + s_[0] * (hi_t[0] - lo_t[0])).item()
        zeta_f= (lo_t[1] + s_[1] * (hi_t[1] - lo_t[1])).item()
        lam_f = (lo_t[2] + s_[2] * (hi_t[2] - lo_t[2])).item()
        sigma_f = float(np.sqrt(zeta_f**2 + lam_f**2))
        sigma_f = max(sigma_f, 0.01)
        rho_f   = float(np.clip(zeta_f / sigma_f, -0.9, -0.1))

    return {
        "v0":      v0_f,
        "zeta":    zeta_f,
        "lambda":  lam_f,
        "sigma":   sigma_f,
        "rho":     rho_f,
        "history": all_hist,
        "elapsed": elapsed,
    }


def compute_confidence_reparameterized(model, v0: float, zeta: float, lam: float,
                                        T_grid, K_grid) -> dict:
    """
    Jacobian column Frobenius norms in real IV space for (v₀, ζ, λ).

    Returns scores normalised to [0,1]; expects all ≥ 0.7 if model works.
    """
    model.eval()
    _load_normalizers()
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)

    params3 = torch.tensor([v0, zeta, lam], dtype=torch.float32, requires_grad=True)

    def _iv_flat(p3):
        """(v₀, ζ, λ) → flat real IV vector."""
        v0_  = p3[0].unsqueeze(0)
        z_   = p3[1].unsqueeze(0)
        l_   = p3[2].unsqueeze(0)
        p6   = _reparam_to_6d(v0_, z_, l_, device)
        return _fno_predict_real_iv(model, p6, spatial).reshape(-1)

    J = torch.autograd.functional.jacobian(
        _iv_flat, params3, create_graph=False, vectorize=False,
    )   # (nT*nK, 3)

    col_norms  = J.pow(2).sum(dim=0).sqrt()            # (3,)
    max_norm   = col_norms.max().clamp(min=1e-12)
    scores     = (col_norms / max_norm).detach().cpu().numpy()

    return {"v0": float(scores[0]), "zeta": float(scores[1]), "lambda": float(scores[2])}


if __name__ == "__main__":

    from fno_model import MirrorPaddedFNO2d

    model   = MirrorPaddedFNO2d()
    T_grid  = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid  = np.linspace(-0.5, 0.5, 11)
    target  = np.random.rand(8, 11) * 0.2 + 0.1
    init_p  = np.array([2.5, 0.08, 0.5, -0.5, 0.08, 0.08])

    print("Testing calibration pipeline with normalizers...")
    params, hist, t = calibrate_parameters(model, target, init_p, T_grid, K_grid)
    print(f"Finished in {t:.3f}s  |  final loss: {hist[-1]:.6f}")
    print(f"Params: {params}")

    scores = compute_confidence_scores(model, params, T_grid, K_grid)
    for name, s in scores.items():
        print(f"  {name:6s}: {s:.3f}")
