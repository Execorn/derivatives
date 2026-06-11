"""
validation.py — Reparameterized calibration accuracy validation suite.

Tests how accurately (v₀, ζ=σρ, λ=σ√(1-ρ²)) can be recovered from noisy IV surfaces
at multiple noise levels.

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python src/validation.py
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d
from calibrate import (
    _make_spatial_input, _fno_predict_real_iv, _load_normalizers,
    calibrate_reparameterized,
)

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

# Sampling bounds for the 3 optimised params (ghost params fixed)
V0_RANGE    = (0.02, 0.12)   # v0
SIGMA_RANGE = (0.15, 0.80)   # sigma (for back-transform)
RHO_RANGE   = (-0.85, -0.15) # rho (for back-transform)


def generate_true_surface(model, v0, sigma, rho, T_grid, K_grid, device):
    """Generate true IV surface from FNO using fixed ghost params."""
    kappa, theta, H = 1.0, 0.08, 0.08
    params = np.array([[kappa, theta, sigma, rho, v0, H]], dtype=np.float32)
    params_t = torch.tensor(params, dtype=torch.float32)
    spatial  = _make_spatial_input(T_grid, K_grid, device)
    with torch.no_grad():
        iv = _fno_predict_real_iv(model, params_t, spatial)
    return iv.cpu().numpy()   # (nT, nK)


def validate_reparameterized_calibration(
    model,
    n_samples: int = 100,
    noise_levels: list = None,
    seed: int = 42,
) -> dict:
    """
    Test parameter recovery accuracy for (v₀, ζ, λ) at multiple noise levels.

    For each noise level:
      1. Sample (v₀, σ, ρ) from uniform training distribution
      2. Compute ζ=σρ, λ=σ√(1-ρ²)
      3. Generate target IV via FNO (truth)
      4. Add Gaussian noise: IV_noisy = IV_true + N(0, noise*|IV_true|)
      5. Calibrate with calibrate_reparameterized()
      6. Report relative parameter recovery errors

    Success: <2% error at 0% noise, <5% error at 1% noise.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02]

    _load_normalizers()
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    # Pre-sample parameter sets (same for all noise levels)
    v0_true    = rng.uniform(*V0_RANGE, size=n_samples)
    sigma_true = rng.uniform(*SIGMA_RANGE, size=n_samples)
    rho_true   = rng.uniform(*RHO_RANGE, size=n_samples)
    zeta_true  = sigma_true * rho_true
    lam_true   = sigma_true * np.sqrt(1.0 - rho_true**2)

    # Clip to calibration bounds
    zeta_true = np.clip(zeta_true, -0.90, -0.01)
    lam_true  = np.clip(lam_true,   0.01,  0.99)

    print(f"\n{'='*76}")
    print(" Reparameterized Calibration Validation: (v₀, ζ=σρ, λ=σ√(1-ρ²))")
    print(f"{'='*76}")
    print(f"  n_samples = {n_samples}")
    print(f"  Ghost params fixed: κ=1.0, θ=0.08, H=0.08")
    print(f"  Noise model: IV_noisy = IV_true + N(0, noise×|IV_true|)\n")

    results = {}

    # Column header
    print(f"  {'Noise':>7} | {'|ζ err|%':>9} | {'|λ err|%':>9} | {'|v₀ err|%':>10} | "
          f"{'t/sample':>9} | {'Converged':>9}")
    print(f"  {'-'*70}")

    for noise in noise_levels:
        zeta_errors = []
        lam_errors  = []
        v0_errors   = []
        times       = []
        n_converged = 0

        for i in range(n_samples):
            # Generate true surface
            iv_true = generate_true_surface(
                model, v0_true[i], sigma_true[i], rho_true[i], T_GRID, K_GRID, device
            )

            # Add noise
            if noise > 0:
                noise_arr = rng.normal(0, noise * np.abs(iv_true), iv_true.shape)
                iv_noisy  = np.maximum(iv_true + noise_arr, 1e-4)
            else:
                iv_noisy = iv_true.copy()

            # Calibrate
            t0 = time.time()
            try:
                res = calibrate_reparameterized(
                    model, iv_noisy, T_GRID, K_GRID, max_iter=100
                )
                t_elapsed = time.time() - t0
                times.append(t_elapsed)

                # Relative errors (avoid divide-by-zero)
                zeta_ref = max(abs(zeta_true[i]), 1e-6)
                lam_ref  = max(abs(lam_true[i]),  1e-6)
                v0_ref   = max(abs(v0_true[i]),   1e-6)

                ze = abs(res["zeta"]   - zeta_true[i]) / zeta_ref * 100
                le = abs(res["lambda"] - lam_true[i])  / lam_ref  * 100
                ve = abs(res["v0"]     - v0_true[i])   / v0_ref   * 100

                zeta_errors.append(ze)
                lam_errors.append(le)
                v0_errors.append(ve)

                if res["history"] and res["history"][-1] < 1e-4:
                    n_converged += 1

            except Exception as e:
                times.append(0.0)
                zeta_errors.append(np.nan)
                lam_errors.append(np.nan)
                v0_errors.append(np.nan)

        ze_arr = np.nanmean(zeta_errors)
        le_arr = np.nanmean(lam_errors)
        ve_arr = np.nanmean(v0_errors)
        t_avg  = np.mean(times)
        conv   = n_converged / n_samples * 100

        results[noise] = {
            "zeta_err_pct":   ze_arr,
            "lambda_err_pct": le_arr,
            "v0_err_pct":     ve_arr,
            "t_per_sample":   t_avg,
            "converged_pct":  conv,
        }

        target_z = 2.0 if noise == 0.0 else 5.0
        target_l = 2.0 if noise == 0.0 else 5.0
        target_v = 5.0 if noise == 0.0 else 10.0
        z_ok = "✓" if ze_arr < target_z else "✗"
        l_ok = "✓" if le_arr < target_l else "✗"
        v_ok = "✓" if ve_arr < target_v else "✗"

        print(f"  {noise*100:>6.1f}% | {ze_arr:>8.2f}{z_ok} | {le_arr:>8.2f}{l_ok} | "
              f"{ve_arr:>9.2f}{v_ok} | {t_avg:>8.3f}s | {conv:>8.1f}%")

    print(f"\n  Success criteria:")
    print(f"    0% noise  : |ζ err| < 2%, |λ err| < 2%, |v₀ err| < 5%")
    print(f"    1%/2% noise: |ζ err| < 5%, |λ err| < 5%, |v₀ err| < 10%")
    print(f"{'='*76}\n")

    return results


def load_model(path: str = "artifacts/models/fno_best.pth"):
    """Load FiLM-FNO v1 model."""
    _load_normalizers()
    model = MirrorPaddedFNO2d()
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Loaded model from {path}")
    else:
        print(f"WARNING: {path} not found — using random weights for syntax check")
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Samples per noise level")
    parser.add_argument("--model", default="artifacts/models/fno_best.pth")
    args = parser.parse_args()

    model   = load_model(args.model)
    results = validate_reparameterized_calibration(model, n_samples=args.n)
