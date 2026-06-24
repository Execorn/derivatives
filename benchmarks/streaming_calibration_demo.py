"""
§5.2 — Real-Time Streaming Calibration Demo
============================================
Simulates a live SPX implied-vol surface feed with smoothly evolving parameters
(Ornstein-Uhlenbeck dynamics) and measures Newton calibration latency per tick.

Usage (from repo root):
    .venv/bin/python benchmarks/streaming_calibration_demo.py [--ticks N] [--noise F]
"""

import sys, os, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fno_model import MirrorPaddedFNO2d
from calibrate import _load_normalizers, _make_spatial_input, _fno_predict_real_iv
from calibrate_fast import calibrate_newton

# ─── Grids ─────────────────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)

# Base point: sigma=0.50, rho=-0.60, v0=0.07, kappa=1.0, theta=0.08, H=0.08
BASE_V0   =  0.07
BASE_ZETA =  0.50 * (-0.60)                      # = -0.30
BASE_LAM  =  0.50 * np.sqrt(1.0 - 0.60**2)      # ≈ 0.40

# OU parameters — each column: (v0, zeta, lam)
OU_SPEED = np.array([3.0, 2.0, 2.0])
OU_VOL   = np.array([0.008, 0.020, 0.020])
OU_MEAN  = np.array([BASE_V0, BASE_ZETA, BASE_LAM])

WEIGHTS = "artifacts/weights/fno_v2_final_prod.pth"


def _load():
    model = MirrorPaddedFNO2d()
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"{WEIGHTS} not found — run from repo root.")
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    _load_normalizers(version="v2")
    return model


def _ou_path(n_ticks, dt, rng):
    """Euler-Maruyama OU path, shape (n_ticks, 3)."""
    path, theta = np.empty((n_ticks, 3)), OU_MEAN.copy()
    for i in range(n_ticks):
        theta = theta + OU_SPEED * (OU_MEAN - theta) * dt + OU_VOL * rng.standard_normal(3) * dt**0.5
        # Hard constraints on parameter domain
        theta[0] = np.clip(theta[0], 0.010, 0.14)   # v0 > 0
        sigma = np.sqrt(theta[1]**2 + theta[2]**2 + 1e-8)
        rho_raw = theta[1] / sigma
        theta[1] = np.clip(rho_raw, -0.89, -0.11) * sigma   # -0.9 < rho < -0.1
        theta[2] = np.clip(theta[2], 0.01, 0.99)
        path[i] = theta.copy()
    return path


def _surface(model, spatial, v0, zeta, lam):
    sigma = max(float(np.sqrt(zeta**2 + lam**2)), 0.01)
    rho   = float(np.clip(zeta / sigma, -0.9, -0.1))
    p6d   = torch.tensor([[1.0, 0.08, sigma, rho, v0, 0.08]], dtype=torch.float32)
    with torch.no_grad():
        return _fno_predict_real_iv(model, p6d, spatial).numpy()


def main():
    ap = argparse.ArgumentParser(description="Streaming calibration demo (§5.2)")
    ap.add_argument("--ticks", type=int,   default=30,    help="Number of ticks")
    ap.add_argument("--noise", type=float, default=0.01,  help="Relative IV noise")
    ap.add_argument("--dt",   type=float, default=1/252,  help="Time step (years)")
    ap.add_argument("--seed", type=int,   default=42)
    args = ap.parse_args()

    model   = _load()
    spatial = _make_spatial_input(T_GRID, K_GRID, device=torch.device("cpu"))
    rng     = np.random.default_rng(args.seed)
    path    = _ou_path(args.ticks, args.dt, rng)

    HDR = ("─" * 130 + "\n"
           f"  §5.2 Streaming Calibration Demo — FiLM-FNO v2 + Newton | "
           f"{args.ticks} ticks | noise={args.noise*100:.1f}% | dt={args.dt:.5f} yr\n"
           "─" * 130)
    print(HDR)
    col = ("  {:>3}  |  {:>6} {:>7} {:>7}  |  {:>6} {:>7} {:>7}  |"
           "  {:>5} {:>5} {:>5}  |  {:>8}  |  {:>7}  | {:>2}it")
    print(col.format("Tick",
          "v₀ᵀ", "ζᵀ", "λᵀ",
          "v₀ᶜ", "ζᶜ", "λᶜ",
          "Δv₀%", "Δζ%", "Δλ%",
          "MSE", "Lat(ms)", "N"))
    print("  " + "-" * 126)

    lats, v0e, ze, le, mses = [], [], [], [], []

    for i, (v0, zeta, lam) in enumerate(path):
        iv_clean  = _surface(model, spatial, v0, zeta, lam)
        noise     = rng.normal(0, args.noise * np.abs(iv_clean), iv_clean.shape)
        iv_noisy  = np.maximum(iv_clean + noise, 1e-4)

        t0  = time.perf_counter()
        res = calibrate_newton(model, iv_noisy, T_GRID, K_GRID, max_iter=15, verbose=False)
        lat = (time.perf_counter() - t0) * 1e3
        lats.append(lat)

        ev0 = abs(res["v0"]     - v0)   / max(abs(v0),   1e-6) * 100
        ez  = abs(res["zeta"]   - zeta) / max(abs(zeta), 1e-6) * 100
        el  = abs(res["lambda"] - lam)  / max(abs(lam),  1e-6) * 100
        v0e.append(ev0); ze.append(ez); le.append(el); mses.append(res["final_mse"])

        print(col.format(
            i+1,
            f"{v0:.4f}", f"{zeta:.4f}", f"{lam:.4f}",
            f"{res['v0']:.4f}", f"{res['zeta']:.4f}", f"{res['lambda']:.4f}",
            f"{ev0:.1f}", f"{ez:.1f}", f"{el:.1f}",
            f"{res['final_mse']:.1e}", f"{lat:.0f}", res["n_iter"]))

    lat = np.array(lats)
    print("\n" + "─" * 130)
    print("  SUMMARY")
    print("─" * 130)
    print(f"  Latency (ms)  : mean={lat.mean():.0f}  median={np.median(lat):.0f}"
          f"  p95={np.percentile(lat,95):.0f}  max={lat.max():.0f}")
    print(f"  Throughput    : {1000/lat.mean():.1f} calib/sec (mean)"
          f"  |  {1000/np.median(lat):.1f} calib/sec (median)")
    print(f"  v₀  |err|%   : mean={np.mean(v0e):.1f}  max={np.max(v0e):.1f}")
    print(f"  ζ   |err|%   : mean={np.mean(ze):.1f}  max={np.max(ze):.1f}")
    print(f"  λ   |err|%   : mean={np.mean(le):.1f}  max={np.max(le):.1f}")
    print(f"  MSE           : mean={np.mean(mses):.2e}  max={np.max(mses):.2e}")

    p95 = np.percentile(lat, 95)
    verdict = (f"  [SUCCESS] Real-time capable: p95={p95:.0f}ms < 1000ms"
               f"  (SPX ticks ~1-2s → {1000/p95:.1f}× headroom)")
    if p95 >= 1000:
        verdict = f"  [WARNING] Borderline: p95={p95:.0f}ms — consider reducing GN iterations"
    print(verdict)
    print("─" * 130)

    # Save result summary
    out = os.path.join(os.path.dirname(__file__), "streaming_demo_results.txt")
    with open(out, "w") as f:
        f.write(f"§5.2 Streaming Calibration Demo — {args.ticks} ticks, noise={args.noise*100:.1f}%\n")
        f.write(f"Latency (ms): mean={lat.mean():.0f} median={np.median(lat):.0f} "
                f"p95={p95:.0f} max={lat.max():.0f}\n")
        f.write(f"Throughput: {1000/lat.mean():.1f} calib/sec\n")
        f.write(f"v0 mean|err|%={np.mean(v0e):.1f}  zeta={np.mean(ze):.1f}  lam={np.mean(le):.1f}\n")
        f.write(verdict.strip() + "\n")
    print(f"\n  Results saved → {out}\n")


if __name__ == "__main__":
    main()
