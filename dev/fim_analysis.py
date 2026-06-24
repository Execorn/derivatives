"""
fim_analysis.py — Fisher Information Matrix analysis for Lifted Rough Heston.

Demonstrates that calibrating in the reparameterized (v₀, ζ=σρ, λ=σ√(1-ρ²)) space
reduces the FIM condition number from ~10⁷–10⁸ (6D) to ~10³–10⁴ (3D).

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python src/fim_analysis.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d
from calibrate import (
    _make_spatial_input, _fno_predict_real_iv, _load_normalizers,
    _reparam_to_6d, _BOUNDS_LOWER_3D, _BOUNDS_UPPER_3D,
    _BOUNDS_LOWER, _BOUNDS_UPPER,
)

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
PARAM_NAMES_6D = ["kappa", "theta", "sigma", "rho", "v0", "H"]
PARAM_NAMES_3D = ["v0", "zeta", "lambda"]


def compute_fim(model, params_6d: np.ndarray, T_grid, K_grid,
                epsilon: float = 1e-4):
    """
    Compute the 6×6 Fisher Information Matrix using 5-point central finite differences.

    FIM[i,j] = Σₖ (∂IV_k/∂θᵢ)(∂IV_k/∂θⱼ)   summed over 88 grid cells

    5-point central difference formula (O(h⁴)):
        f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)

    Returns: (FIM_6x6, eigenvalues_sorted_asc, condition_number)
    """
    model.eval()
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)

    n_params = len(params_6d)
    lo = _BOUNDS_LOWER.numpy()
    hi = _BOUNDS_UPPER.numpy()

    def iv_flat(p):
        """(6,) np → (88,) np flat IV."""
        p_t = torch.tensor(p, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            iv = _fno_predict_real_iv(model, p_t, spatial)
        return iv.cpu().numpy().reshape(-1)

    # Compute per-parameter gradient via 5-point FD
    grads = np.zeros((88, n_params), dtype=np.float64)

    for i in range(n_params):
        h = epsilon * (hi[i] - lo[i])   # scale h by parameter range
        h = max(h, 1e-6)

        def perturb(delta):
            p = params_6d.copy()
            p[i] = np.clip(p[i] + delta * h, lo[i], hi[i])
            return iv_flat(p)

        f_p2 = perturb(+2)
        f_p1 = perturb(+1)
        f_m1 = perturb(-1)
        f_m2 = perturb(-2)

        grads[:, i] = (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12.0 * h)

    FIM = grads.T @ grads   # (6, 6)

    eigvals = np.linalg.eigvalsh(FIM)   # ascending
    eigvals = np.clip(eigvals, 0, None)  # numerical safety
    cond = float(eigvals[-1] / max(eigvals[0], 1e-20))

    return FIM, eigvals, cond


def compute_fim_3d(model, v0: float, zeta: float, lam: float,
                   T_grid, K_grid, epsilon: float = 1e-4):
    """
    Compute the 3×3 FIM in the reparameterized (v₀, ζ, λ) space.

    Returns: (FIM_3x3, eigenvalues_sorted_asc, condition_number)
    """
    model.eval()
    device = next(model.parameters()).device
    spatial = _make_spatial_input(T_grid, K_grid, device)

    params3 = np.array([v0, zeta, lam])
    lo3 = _BOUNDS_LOWER_3D.numpy()
    hi3 = _BOUNDS_UPPER_3D.numpy()

    def iv_flat_3d(p3):
        v0_  = torch.tensor([p3[0]], dtype=torch.float32)
        z_   = torch.tensor([p3[1]], dtype=torch.float32)
        l_   = torch.tensor([p3[2]], dtype=torch.float32)
        p6   = _reparam_to_6d(v0_, z_, l_, device)
        with torch.no_grad():
            iv = _fno_predict_real_iv(model, p6, spatial)
        return iv.cpu().numpy().reshape(-1)

    grads3 = np.zeros((88, 3), dtype=np.float64)
    for i in range(3):
        h = epsilon * (hi3[i] - lo3[i])
        h = max(h, 1e-6)

        def perturb3(delta, _i=i):
            p = params3.copy()
            p[_i] = np.clip(p[_i] + delta * h, lo3[_i], hi3[_i])
            return iv_flat_3d(p)

        f_p2 = perturb3(+2)
        f_p1 = perturb3(+1)
        f_m1 = perturb3(-1)
        f_m2 = perturb3(-2)

        grads3[:, i] = (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12.0 * h)

    FIM3 = grads3.T @ grads3
    eigvals3 = np.linalg.eigvalsh(FIM3)
    eigvals3 = np.clip(eigvals3, 0, None)
    cond3 = float(eigvals3[-1] / max(eigvals3[0], 1e-20))

    return FIM3, eigvals3, cond3


def compare_fim_spaces(model, n_samples: int = 20,
                       T_grid=T_GRID, K_grid=K_GRID):
    """
    Sample n_samples random parameter vectors and compare:
    - 6D FIM condition number (κ,θ,σ,ρ,v₀,H)
    - 3D FIM condition number (v₀,ζ,λ)

    Expected: 6D cond ~10⁷–10⁸, 3D cond ~10³–10⁴ (4+ orders of magnitude improvement).
    """
    rng = np.random.default_rng(seed=42)
    lo6 = _BOUNDS_LOWER.numpy()
    hi6 = _BOUNDS_UPPER.numpy()

    conds_6d = []
    conds_3d = []

    print(f"\n{'='*72}")
    print(" Fisher Information Matrix Analysis: 6D vs 3D Parameter Space")
    print(f"{'='*72}")
    print(f"  {'Sample':>6}  {'Cond(6D)':>12}  {'Cond(3D)':>12}  {'Ratio':>10}")
    print(f"  {'-'*50}")

    for k in range(n_samples):
        # Sample uniformly in 6D space
        p6 = lo6 + rng.random(6) * (hi6 - lo6)

        # Derive 3D params: v0 direct, zeta=sigma*rho, lambda=sigma*sqrt(1-rho^2)
        sigma, rho, v0 = p6[2], p6[3], p6[4]
        zeta = sigma * rho
        lam  = sigma * np.sqrt(1 - rho**2)
        zeta = np.clip(zeta, -0.90, -0.01)
        lam  = np.clip(lam,   0.01,  0.99)

        _, _, c6 = compute_fim(model, p6, T_grid, K_grid)
        _, _, c3 = compute_fim_3d(model, v0, zeta, lam, T_grid, K_grid)

        conds_6d.append(c6)
        conds_3d.append(c3)
        ratio = c6 / max(c3, 1.0)
        print(f"  {k+1:>6}  {c6:>12.2e}  {c3:>12.2e}  {ratio:>10.1f}×")

    c6 = np.array(conds_6d)
    c3 = np.array(conds_3d)

    print(f"\n{'='*72}")
    print(f"  {'Statistic':>12}  {'6D FIM cond':>14}  {'3D FIM cond':>14}  {'Ratio':>10}")
    print(f"  {'-'*56}")
    for stat, fn in [("Mean", np.mean), ("Median", np.median),
                     ("Min", np.min), ("Max", np.max)]:
        print(f"  {stat:>12}  {fn(c6):>14.2e}  {fn(c3):>14.2e}  {fn(c6)/max(fn(c3),1):>10.1f}×")
    print(f"{'='*72}\n")

    print("Interpretation:")
    print(f"  6D condition number (mean): {np.mean(c6):.2e}")
    print(f"  3D condition number (mean): {np.mean(c3):.2e}")
    print(f"  → Reparameterization reduces FIM cond by {np.mean(c6)/np.mean(c3):.0f}× on average")
    print(f"  → 3D space is {'well-conditioned' if np.mean(c3) < 1e5 else 'still ill-conditioned'}")

    return {"conds_6d": conds_6d, "conds_3d": conds_3d}


def load_model(path: str = "artifacts/models/fno_best.pth"):
    """Load FiLM-FNO v1 model."""
    _load_normalizers()
    model = MirrorPaddedFNO2d()
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from {path}")
    else:
        print(f"WARNING: {path} not found — using random weights")
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of samples")
    parser.add_argument("--model", default="artifacts/models/fno_best.pth")
    args = parser.parse_args()

    model = load_model(args.model)
    results = compare_fim_spaces(model, n_samples=args.n)
