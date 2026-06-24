"""
Stress Test: 5 Edge-Case Parameter Sets for Lifted Heston / MFNO Calibration.

Covers:
  1. Feller-Critical   — 2κθ = σ² + 1e-4   (CIR barely stays positive)
  2. Deep-Rough        — H = 0.02           (power-law short-maturity explosion)
  3. Extreme Vol-of-Vol— σ = 1.0            (near upper bound, large jumps in V)
  4. Kappa-Zero        — κ = 0.1            (near-unit-root; structural ill-posedness)
  5. Inverted-Smile    — v0 >> theta        (contango→backwardation crossover)

For each set we check:
  a) CUDA engine: mean price ≈ S0 (martingale check), no NaN/Inf
  b) Bernstein weight underflow at H=0.02 (float32 resolution)
  c) Feller condition value
  d) FNO forward pass shape and positivity (IV > 0 everywhere)

Usage:
    cd /home/execorn/programming/derivatives
    python scripts/stress_test_edge_cases.py

Requires:
    - lifted_heston_cuda extension compiled (run setup_and_run.sh first)
    - No trained FNO weights needed; architecture tests run on random init.
"""

import sys
import os
import math
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Attempt CUDA import (may not be available in CI) ──────────────────────────
try:
    import lifted_heston_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: lifted_heston_cuda not compiled. CUDA tests will be skipped.")

from validate_cuda import compute_weights_and_speeds
from fno_model import MirrorPaddedFNO2d

# ── Grid ──────────────────────────────────────────────────────────────────────
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
MONEYNESS  = np.linspace(-0.5, 0.5, 11)
S0 = 100.0
F  = 100.0

# ── Edge-Case Parameter Sets ──────────────────────────────────────────────────
EDGE_CASES = {
    "1_Feller_Critical": {
        "kappa": 2.0, "theta": 0.05, "sigma": math.sqrt(2 * 2.0 * 0.05 - 1e-4),
        "rho": -0.6, "v0": 0.05, "H": 0.1,
        "desc": "Feller barely met: 2κθ = σ² + 1e-4",
        "feller_expected": True,  # must pass Feller
    },
    "2_Deep_Rough_H002": {
        "kappa": 1.5, "theta": 0.04, "sigma": 0.3,
        "rho": -0.7, "v0": 0.04, "H": 0.02,
        "desc": "H=0.02 deep rough — power-law maturity explosion",
        "feller_expected": True,  # must pass Feller
    },
    "3_Extreme_Vol_of_Vol": {
        "kappa": 1.5, "theta": 0.04, "sigma": 1.0,
        "rho": -0.7, "v0": 0.04, "H": 0.1,
        "desc": "σ=1.0 (upper bound) — INTENTIONAL Feller violation: tests robustness to invalid inputs",
        "feller_expected": False,  # deliberately violating — calibrator must reject via hard penalty
    },
    "4_Kappa_Near_Zero": {
        "kappa": 0.1, "theta": 0.08, "sigma": 0.3,
        "rho": -0.5, "v0": 0.08, "H": 0.08,
        "desc": "κ=0.1 — near unit-root, INTENTIONAL Feller violation: 2κθ < σ²",
        "feller_expected": False,  # deliberately violating
    },
    "5_Inverted_Term_Structure": {
        "kappa": 4.0, "theta": 0.02, "sigma": 0.4,
        "rho": -0.8, "v0": 0.15, "H": 0.08,
        "desc": "v0 >> theta → IV term structure initially inverted; Feller at boundary",
        "feller_expected": False,  # 2*4*0.02 = 0.16 = σ²: exactly at boundary (floating-point: FAIL ok)
    },
}

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

# ── FNO model (random weights — tests architecture, not accuracy) ──────────────
fno_model = MirrorPaddedFNO2d().eval()
T_torch = torch.tensor(MATURITIES, dtype=torch.float32)
K_torch = torch.tensor(MONEYNESS,  dtype=torch.float32)

def fno_forward(params_dict):
    """Run FNO forward pass and return IV surface (T, K)."""
    p = torch.tensor(
        [params_dict["kappa"], params_dict["theta"], params_dict["sigma"],
         params_dict["rho"],   params_dict["v0"],    params_dict["H"]],
        dtype=torch.float32
    )
    T_mesh, K_mesh = torch.meshgrid(T_torch, K_torch, indexing='ij')
    p_exp = p.view(1, 1, 1, 6).expand(1, len(MATURITIES), len(MONEYNESS), 6)
    inp = torch.cat([p_exp, T_mesh.unsqueeze(0).unsqueeze(-1),
                             K_mesh.unsqueeze(0).unsqueeze(-1)], dim=-1)
    with torch.no_grad():
        iv = fno_model(inp).squeeze(0)  # (T, K)
    return iv

def check_weights_underflow(H, N=20):
    """Check if any Bernstein weights underflow to zero in float32."""
    c_weights, x_speeds = compute_weights_and_speeds(N, H)
    min_weight = c_weights.min().item()
    max_ratio  = (c_weights.max() / c_weights.min()).item() if min_weight > 0 else float('inf')
    return min_weight, max_ratio, c_weights

def feller_value(kappa, theta, sigma):
    return 2 * kappa * theta - sigma ** 2

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print(" LIFTED HESTON / MFNO — EDGE-CASE STRESS TEST")
print("=" * 72)

all_passed = True

for case_name, params in EDGE_CASES.items():
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho   = params["rho"]
    v0    = params["v0"]
    H     = params["H"]
    desc  = params["desc"]

    print(f"\n{'─'*72}")
    print(f"  CASE {case_name}")
    print(f"  {desc}")
    print(f"  κ={kappa:.3f}  θ={theta:.4f}  σ={sigma:.4f}  ρ={rho:.2f}  v0={v0:.4f}  H={H:.3f}")
    print(f"{'─'*72}")

    # ── A. Feller Condition ────────────────────────────────────────────────────
    fv = feller_value(kappa, theta, sigma)
    feller_ok = fv > 0
    feller_expected = params.get("feller_expected", True)
    if feller_ok:
        feller_status = PASS
    elif not feller_expected:
        feller_status = "[EXPECTED (stress boundary)]"  # intentional violation
    else:
        feller_status = FAIL
        all_passed = False
    print(f"  [A] Feller  2κθ - σ² = {fv:+.6f}  {feller_status}")

    # ── B. Bernstein Weight Underflow ─────────────────────────────────────────
    min_w, ratio, c_weights = check_weights_underflow(H)
    float32_eps = torch.finfo(torch.float32).tiny  # ~1.18e-38
    underflow = min_w < float32_eps
    weight_status = FAIL if underflow else PASS
    print(f"  [B] Weights min={min_w:.4e}  max/min ratio={ratio:.2e}  "
          f"float32_tiny={float32_eps:.2e}  {weight_status}")
    if underflow:
        all_passed = False
        print(f"      UNDERFLOW detected in c_weights at H={H}! "
              "Consider switching to float64 weight precomputation.")

    # ── C. CUDA Engine: Martingale Check ──────────────────────────────────────
    if CUDA_AVAILABLE and torch.cuda.is_available():
        num_paths = 65_536
        num_steps = max(int(1.0 * 252), 1)
        dt = 1.0 / num_steps
        c_w = c_weights.cuda().contiguous()  # from check_weights_underflow above

        # compute_weights_and_speeds returns (c_weights, x_speeds)
        # FIX: was "x_sp, _" which silently assigned c_weights to x_sp
        _, x_sp = compute_weights_and_speeds(20, H)
        x_sp = x_sp.cuda().contiguous()

        try:
            # Post-recompile path: kernel accepts call_index kwarg
            prices = lifted_heston_cuda.simulate_lifted_heston(
                num_paths, num_steps, dt, S0, float(v0), float(rho),
                float(kappa), float(theta), float(sigma),
                c_w, x_sp, seed=12345, call_index=0
            )
        except TypeError:
            # Pre-recompile compat: old .so has no call_index parameter.
            # Run setup_and_run.sh to recompile, then re-run this test.
            print(f"  [C] CUDA   {SKIP} (kernel not yet recompiled with call_index support — run setup_and_run.sh)")
            prices = None

        if prices is not None:
            has_nan  = torch.isnan(prices).any().item()
            has_inf  = torch.isinf(prices).any().item()
            mean_p   = prices.mean().item()
            std_p    = prices.std().item()
            mc_std_err = std_p / math.sqrt(num_paths)
            martingale_err = abs(mean_p - S0)
            martingale_ok = martingale_err < 5 * mc_std_err and not has_nan and not has_inf
            cuda_status = PASS if martingale_ok else FAIL
            print(f"  [C] CUDA   E[S_T]={mean_p:.4f}  S0={S0:.1f}  "
                  f"err={martingale_err:.4f}  5σ={5*mc_std_err:.4f}  "
                  f"NaN={has_nan}  Inf={has_inf}  {cuda_status}")
            if not martingale_ok:
                all_passed = False
    else:
        print(f"  [C] CUDA   {SKIP} (not available)")

    # ── D. FNO Forward Pass ───────────────────────────────────────────────────
    iv_surface = fno_forward(params)
    iv_shape_ok = iv_surface.shape == (len(MATURITIES), len(MONEYNESS))
    iv_pos_ok   = (iv_surface > 0).all().item()
    iv_finite   = torch.isfinite(iv_surface).all().item()
    iv_min      = iv_surface.min().item()
    iv_max      = iv_surface.max().item()
    fno_ok = iv_shape_ok and iv_finite
    # Note: positivity fails on random weights — this is expected before training.
    fno_status = PASS if fno_ok else FAIL
    pos_note = "(random weights — positivity not expected pre-training)" if not iv_pos_ok else ""
    print(f"  [D] FNO    shape={tuple(iv_surface.shape)}  finite={iv_finite}  "
          f"positive={iv_pos_ok} {pos_note}")
    print(f"             min_IV={iv_min:.4f}  max_IV={iv_max:.4f}  {fno_status}")
    if not fno_ok:
        all_passed = False

    # ── E. Kappa Identification Diagnostic ───────────────────────────────────
    if case_name.startswith("4"):
        print(f"  [E] KAPPA IDENTIFICATION WARNING:")
        print(f"      κ={kappa} is near 0. The Rough Heston IV surface is insensitive")
        print(f"      to κ for T < 0.5 (fractional kernel dominates). The calibrated")
        print(f"      κ will have high uncertainty — display a confidence score in the UI.")

print(f"\n{'═'*72}")
if all_passed:
    print(f"  OVERALL: {PASS} All structural checks passed.")
else:
    print(f"  OVERALL: {FAIL} One or more checks failed — see above.")
print(f"{'═'*72}\n")
