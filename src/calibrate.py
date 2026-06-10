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
