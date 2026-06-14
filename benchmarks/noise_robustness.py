"""
noise_robustness.py — Newton vs L-BFGS calibration noise-robustness study.

Compares two calibrators on the FNO v2 model (R²=0.9991):
  • Newton-Raphson   (calibrate_newton,  jacfwd — 3 JVPs per Jacobian)
  • L-BFGS           (calibrate_reparameterized, scipy.optimize)

IV target: IV_noisy = IV_true + N(0, noise × |IV_true|)  clipped at 1e-4.
Parameters sampled via Sobol over the 3D reparameterized space (v₀, ζ, λ).

Usage:
    .venv/bin/python benchmarks/noise_robustness.py [--n 30] [--seed 42]

Output:  benchmarks/noise_robustness_results.txt
"""
import os, sys, time, argparse
import numpy as np
import torch
from scipy.stats import qmc

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Patch v2 normalizer paths BEFORE importing calibrate functions
import calibrate as _cal_mod
_cal_mod._PARAM_NORM_PATH = os.path.join(ROOT, 'artifacts', 'models', 'param_normalizer_v2.npz')
_cal_mod._IV_NORM_PATH    = os.path.join(ROOT, 'artifacts', 'models', 'iv_normalizer_v2.npz')
_cal_mod._param_norm = None
_cal_mod._iv_norm    = None

from calibrate import (
    _load_normalizers, _make_spatial_input,
    _fno_predict_real_iv, calibrate_reparameterized,
)
from calibrate_fast import calibrate_newton, _reparam_to_6d
from fno_model import MirrorPaddedFNO2d

T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

BOUNDS_LO = np.array([0.01, -0.90, 0.05])
BOUNDS_HI = np.array([0.15, -0.01, 0.95])
NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02]


def _load_v2_model(device):
    path = os.path.join(ROOT, 'artifacts', 'weights', 'fno_v2_final_prod.pth')
    m = MirrorPaddedFNO2d().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


def _generate_targets(model, params, spatial, device):
    out = []
    for row in params:
        v0 = torch.tensor([[row[0]]], dtype=torch.float32, device=device)
        ze = torch.tensor([[row[1]]], dtype=torch.float32, device=device)
        la = torch.tensor([[row[2]]], dtype=torch.float32, device=device)
        p6 = _reparam_to_6d(v0, ze, la, device)
        with torch.no_grad():
            iv = _fno_predict_real_iv(model, p6, spatial).cpu().numpy()
        out.append(iv)
    return np.array(out)


def _rel_err_pct(est, true):
    return abs(est - true) / max(abs(true), 1e-6) * 100.0


def run(n_samples=30, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print("Loading FNO v2 + v2 normalizers ...", end=' ', flush=True)
    _load_normalizers()
    model   = _load_v2_model(device)
    spatial = _make_spatial_input(T_GRID, K_GRID, device)
    print("done")

    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    params  = qmc.scale(sampler.random(n_samples), BOUNDS_LO, BOUNDS_HI)

    print(f"Generating {n_samples} IV surfaces ...", end=' ', flush=True)
    iv_true = _generate_targets(model, params, spatial, device)
    print("done\n")

    rng = np.random.default_rng(seed)
    W   = 72
    print("=" * W)
    print(" Newton-Raphson vs L-BFGS — Calibration Noise Robustness")
    print(f" FNO v2 (R²=0.9991, N=40, N_cos=128)  |  n={n_samples} Sobol samples")
    print("=" * W)
    print(f"{'Noise':>7} | {'Method':<8} | {'|ζ| err%':>8} {'|λ| err%':>8} "
          f"{'|v₀| err%':>9} | {'Conv%':>6} | {'ms/smp':>7}")
    print("-" * W)

    all_results = {}
    for noise in NOISE_LEVELS:
        row = {}
        for method in ('newton', 'lbfgs'):
            ze_e, la_e, v0_e, ts, convs = [], [], [], [], []
            for i in range(n_samples):
                iv_n = iv_true[i].copy()
                if noise > 0:
                    iv_n = np.maximum(
                        iv_n + rng.normal(0, noise * np.abs(iv_n), iv_n.shape), 1e-4)
                t0 = time.perf_counter()
                try:
                    if method == 'newton':
                        res  = calibrate_newton(model, iv_n, T_GRID, K_GRID,
                                                max_iter=20, n_restarts=3, verbose=False)
                        conv = res['final_mse'] < 1e-4
                    else:
                        res  = calibrate_reparameterized(model, iv_n, T_GRID, K_GRID,
                                                         max_iter=100)
                        h    = res.get('history', [])
                        conv = bool(h and h[-1] < 1e-4)
                    ze_e.append(_rel_err_pct(res['zeta'],   params[i, 1]))
                    la_e.append(_rel_err_pct(res['lambda'], params[i, 2]))
                    v0_e.append(_rel_err_pct(res['v0'],     params[i, 0]))
                except Exception:
                    ze_e.append(np.nan); la_e.append(np.nan); v0_e.append(np.nan)
                    conv = False
                ts.append((time.perf_counter() - t0) * 1e3)
                convs.append(conv)

            ze_m = np.nanmean(ze_e); la_m = np.nanmean(la_e)
            v0_m = np.nanmean(v0_e); t_m  = np.mean(ts)
            c_pct = np.mean(convs) * 100
            row[method] = dict(ze=ze_m, la=la_m, v0=v0_m, t=t_m, conv=c_pct)

            lbl = 'Newton' if method == 'newton' else 'L-BFGS'
            print(f"{noise*100:>6.1f}% | {lbl:<8} | {ze_m:>8.2f} {la_m:>8.2f} "
                  f"{v0_m:>9.2f} | {c_pct:>6.1f} | {t_m:>7.1f}")
        all_results[noise] = row
        print("-" * W)

    out = os.path.join(os.path.dirname(__file__), 'noise_robustness_results.txt')
    with open(out, 'w') as f:
        f.write("Newton vs L-BFGS Noise Robustness — FNO v2 (R²=0.9991)\n")
        f.write(f"n={n_samples} Sobol samples/level  seed={seed}\n{'='*72}\n")
        f.write(f"{'Noise':>7} | {'Method':<8} | {'ze%':>8} {'la%':>8} {'v0%':>9}"
                f" | {'Conv%':>6} | {'ms':>7}\n{'-'*72}\n")
        for noise in NOISE_LEVELS:
            for method, lbl in [('newton','Newton'), ('lbfgs','L-BFGS')]:
                r = all_results[noise][method]
                f.write(f"{noise*100:>6.1f}% | {lbl:<8} | {r['ze']:>8.2f} "
                        f"{r['la']:>8.2f} {r['v0']:>9.2f} | {r['conv']:>6.1f} "
                        f"| {r['t']:>7.1f}\n")
    print(f"\n  Saved: {out}")
    return all_results


if __name__ == '__main__':
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument('--n',    type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    run(a.n, a.seed)
