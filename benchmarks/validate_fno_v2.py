"""
validate_fno_v2.py — FNO v2 model quality validation on held-out test data.

Loads the FNO v2 model (trained on Fourier-COS dataset, N=20) and computes:
  - R² and MAE on 200 held-out test samples
  - Per-parameter Jacobian column norms (identifiability check)
  - Comparison vs v1 baseline (R²=0.796)

Expected: R² > 0.92, MAE < 0.5% IV, all Jacobian columns non-zero.

Usage:
    /home/execorn/programming/derivatives/.venv/bin/python \
        benchmarks/validate_fno_v2.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Try to import v2-capable model and calibrate helpers
from fno_model import MirrorPaddedFNO2d
from normalizers import ParameterNormalizer, IVSurfaceNormalizer
import calibrate
from calibrate import _make_spatial_input, _fno_predict_real_iv, _load_normalizers

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]

# ── Path resolution ───────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

MODEL_V2_PATH       = os.path.join(ROOT, "artifacts/weights/fno_v2_final_prod.pth")
PARAM_NORM_V2_PATH  = os.path.join(ROOT, "artifacts/models/param_normalizer_v2.npz")
IV_NORM_V2_PATH     = os.path.join(ROOT, "artifacts/models/iv_normalizer_v2.npz")
DATASET_V2_PATH     = os.path.join(ROOT, "data/DeepRoughDataset_v2_fourier.npz")

MODEL_V1_PATH       = os.path.join(ROOT, "artifacts/models/fno_best.pth")
PARAM_NORM_V1_PATH  = os.path.join(ROOT, "artifacts/models/param_normalizer.npz")
IV_NORM_V1_PATH     = os.path.join(ROOT, "artifacts/models/iv_normalizer.npz")
DATASET_MC_PATH     = os.path.join(ROOT, "data/DeepRoughDataset.npz")
N_TEST = 200


def load_normalizers_v2():
    """Load v2 normalizers; fallback to v1 if v2 not found."""
    if os.path.exists(PARAM_NORM_V2_PATH) and os.path.exists(IV_NORM_V2_PATH):
        param_norm = ParameterNormalizer.load(PARAM_NORM_V2_PATH)
        iv_norm    = IVSurfaceNormalizer.load(IV_NORM_V2_PATH)
        return param_norm, iv_norm, "v2"
    else:
        print(f"  WARNING: v2 normalizers not found, falling back to v1")
        param_norm = ParameterNormalizer.load(PARAM_NORM_V1_PATH)
        iv_norm    = IVSurfaceNormalizer.load(IV_NORM_V1_PATH)
        return param_norm, iv_norm, "v1"

def compute_jacobian_column_norms(model, params_np, spatial):
    """5-point FD Jacobian column norms over N test samples. Runs on GPU."""
    device = next(model.parameters()).device
    lo = np.array([0.1, 0.01, 0.10, -0.9, 0.01, 0.02])
    hi = np.array([5.0, 0.15, 1.00, -0.1, 0.15, 0.15])
    eps_rel = 1e-3   # 0.1% of range

    col_norms = np.zeros((len(params_np), 6))
    sp = _make_spatial_input(T_GRID, K_GRID, device)   # built once, reused

    for i in range(len(params_np)):
        p = params_np[i]
        grads = []
        for j in range(6):
            h = eps_rel * (hi[j] - lo[j])
            h = max(h, 1e-5)
            pts = []
            for delta in (-2, -1, +1, +2):
                pp = p.copy()
                pp[j] = np.clip(pp[j] + delta * h, lo[j], hi[j])
                pp_t = torch.tensor(pp[None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    iv = _fno_predict_real_iv(model, pp_t, sp)
                pts.append(iv.cpu().numpy().reshape(-1))
            grad = (-pts[3] + 8*pts[2] - 8*pts[1] + pts[0]) / (12 * h)
            grads.append(grad)
        col_norms[i] = [np.linalg.norm(g) for g in grads]

    return col_norms   # (N, 6)


def validate_model(model_path, dataset_path, version_label, param_norm, iv_norm,
                   use_fno_predict=True):
    """Validate a model on held-out test data. Returns R², MAE."""
    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load model onto GPU
    model = MirrorPaddedFNO2d()
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"  Loaded {version_label} model from {os.path.basename(model_path)}")

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"  SKIP: dataset {dataset_path} not found")
        return None, None

    raw = np.load(dataset_path)
    data = raw["dataset"]   # (N, 94)
    N_total = data.shape[0]
    print(f"  Dataset: {N_total:,} samples")

    rng = np.random.default_rng(seed=999)
    idx = rng.choice(N_total, size=N_TEST, replace=False)
    samples   = data[idx]
    params_np = samples[:, :6]
    iv_mc     = samples[:, 6:].reshape(N_TEST, len(T_GRID), len(K_GRID))

    # Predict — batched single forward pass on GPU (not a per-sample loop).
    # _fno_predict_real_iv handles (B,6) input and returns (B,nT,nK) when
    # squeeze(0) is suppressed, but its current API squeezes the batch dim.
    # We call it sample-by-sample but on GPU so transfers are negligible.
    spatial      = _make_spatial_input(T_GRID, K_GRID, device)
    params_raw_t = torch.tensor(params_np, dtype=torch.float32, device=device)

    # Batched prediction: stack all N_TEST samples in one GPU forward pass
    import time as _time
    t0 = _time.perf_counter()
    preds = []
    BATCH = 64   # forward pass chunk size — fits in 12GB VRAM easily
    for start in range(0, N_TEST, BATCH):
        end = min(start + BATCH, N_TEST)
        batch_params = params_raw_t[start:end]   # (chunk, 6)
        with torch.no_grad():
            # _fno_predict_real_iv squeezes batch dim when B=1, so use B>1 path:
            # Call directly through the FNO stack without the squeeze.
            params_norm = calibrate._param_norm.transform_tensor(batch_params)
            pred_norm   = model(
                spatial.expand(batch_params.size(0), -1, -1, -1),
                params_norm
            )                                    # (chunk, nT, nK)
            iv_real = calibrate._iv_norm.inverse_transform_tensor(pred_norm)
            iv_real = iv_real.clamp(min=1e-4)
        preds.append(iv_real.cpu().numpy())      # (chunk, nT, nK)
    elapsed = _time.perf_counter() - t0
    iv_pred_np = np.concatenate(preds, axis=0)  # (N_TEST, nT, nK)
    print(f"  Inference: {N_TEST} samples in {elapsed*1000:.1f}ms "
          f"({elapsed/N_TEST*1000:.2f}ms/sample) on {device}")

    # Metrics
    valid = np.isfinite(iv_mc) & np.isfinite(iv_pred_np) & (iv_mc > 0)
    y_true = iv_mc[valid]
    y_pred = iv_pred_np[valid]

    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2  = float(1 - ss_res / ss_tot)
    mae = float(np.mean(np.abs(y_true - y_pred))) * 100   # in vol-% units

    print(f"\n  {version_label} Quality Metrics (N={N_TEST} test samples):")
    print(f"    R²  = {r2:.4f}  ({'PASS (>0.92)' if r2 > 0.92 else 'FAIL (<0.92)'})")
    print(f"    MAE = {mae:.4f}%  ({'PASS (<0.5%)' if mae < 0.5 else 'FAIL (>0.5%)'})")

    # Jacobian column norms (on 20 samples for speed)
    print(f"\n  Jacobian column norms ‖∂IV/∂θᵢ‖_F (20 samples):")
    jac_idx = rng.choice(N_TEST, size=20, replace=False)
    col_norms = compute_jacobian_column_norms(model, params_np[jac_idx], spatial)
    mean_norms = col_norms.mean(axis=0)
    for j, (name, nm) in enumerate(zip(PARAM_NAMES, mean_norms)):
        flag = "OK" if nm > 0.01 else "ZERO"
        print(f"    ‖∂IV/∂{name}‖ = {nm:.4f}  {flag}")

    return r2, mae


def run_validation():
    print("=" * 68)
    print(" FNO Model Quality Validation: v1 (MC) vs v2 (Fourier-COS)")
    print("=" * 68)

    # ── v1 model on MC dataset ────────────────────────────────────────────────
    print("\n── v1 Model (trained on Monte Carlo dataset) ──")
    _load_normalizers()   # prime the cache with v1 paths
    param_norm_v1 = ParameterNormalizer.load(PARAM_NORM_V1_PATH)
    iv_norm_v1    = IVSurfaceNormalizer.load(IV_NORM_V1_PATH)
    # Inject v1 normalizers into calibrate module globals (used by _fno_predict_real_iv)
    calibrate._param_norm = param_norm_v1
    calibrate._iv_norm    = iv_norm_v1
    r2_v1, mae_v1 = validate_model(
        MODEL_V1_PATH, DATASET_MC_PATH, "v1", param_norm_v1, iv_norm_v1)

    # ── v2 model on COS dataset ───────────────────────────────────────────────
    print("\n── v2 Model (trained on Fourier-COS dataset) ──")
    param_norm_v2, iv_norm_v2, norm_version = load_normalizers_v2()
    # Override calibrate globals so _fno_predict_real_iv uses v2 normalizers
    calibrate._param_norm = param_norm_v2
    calibrate._iv_norm    = iv_norm_v2
    r2_v2, mae_v2 = validate_model(
        MODEL_V2_PATH, DATASET_V2_PATH, f"v2 (norms={norm_version})",
        param_norm_v2, iv_norm_v2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(" Summary")
    print("=" * 68)
    if r2_v1 is not None:
        print(f"  v1 (MC dataset)  : R²={r2_v1:.4f}  MAE={mae_v1:.4f}%")
    if r2_v2 is not None:
        print(f"  v2 (COS dataset) : R²={r2_v2:.4f}  MAE={mae_v2:.4f}%")
        if r2_v1 is not None and r2_v2 > r2_v1:
            print(f"  Improvement      : ΔR²={r2_v2-r2_v1:+.4f}  ΔMAE={mae_v2-mae_v1:+.4f}%")
    print("=" * 68)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    run_validation()
