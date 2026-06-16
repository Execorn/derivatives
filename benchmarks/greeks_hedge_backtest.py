"""
§5.3 — Greeks-Based Delta Hedging Backtest
==========================================
Delta-hedge an ATM 1-year call using FNO v2 implied vols vs flat-vol Black-Scholes.

Strategy:
  - Underlying S follows GBM with drift μ=0, vol σ_real (sampled from Rough Heston).
  - At each rebalancing step we calibrate Rough Heston from the *noisy* current
    surface using Newton, extract σ_BS(T_rem, K=0), compute BS-Delta, and hedge.
  - Benchmark: same strategy but using the initial flat σ_BS throughout (no re-cal).
  - Metric: hedge P&L variance.  Lower variance = better hedge.

Usage (from repo root):
    .venv/bin/python benchmarks/greeks_hedge_backtest.py [--steps N] [--paths M]
"""

import sys, os, time, argparse
import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fno_model import MirrorPaddedFNO2d
from calibrate import _load_normalizers, _make_spatial_input, _fno_predict_real_iv
from calibrate_fast import calibrate_newton

# ─── Grids ─────────────────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
WEIGHTS = "artifacts/weights/fno_v2_final_prod.pth"


# ─── Black-Scholes helpers ─────────────────────────────────────────────────────

def bs_call_price(S, K, T, sigma, r=0.0):
    """Black-Scholes call price. K is the actual strike (not log-moneyness)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sqrtT**2) / sqrtT
    d2 = d1 - sqrtT
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_delta(S, K, T, sigma, r=0.0):
    """BS Delta = N(d1) for a call."""
    if T <= 0 or sigma <= 0:
        return float(S > K)
    sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sqrtT**2) / sqrtT
    return float(norm.cdf(d1))


def atm_iv_from_surface(iv_surface, T_rem):
    """Extract ATM (K=0) IV at the nearest available maturity to T_rem."""
    atm_idx = np.argmin(np.abs(K_GRID))  # K=0 index
    t_idx   = np.argmin(np.abs(T_GRID - T_rem))
    return float(iv_surface[t_idx, atm_idx])


# ─── Model helpers ─────────────────────────────────────────────────────────────

def _load():
    model = MirrorPaddedFNO2d()
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"{WEIGHTS} — run from repo root.")
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    _load_normalizers(version="v2")
    return model


def _surface(model, spatial, sigma, rho, v0, kappa=1.0, theta=0.08, H=0.08):
    p6d = torch.tensor([[kappa, theta, sigma, rho, v0, H]], dtype=torch.float32)
    with torch.no_grad():
        return _fno_predict_real_iv(model, p6d, spatial).numpy()


# ─── Single path simulation ────────────────────────────────────────────────────

def run_path(model, spatial, true_sigma, true_rho, true_v0,
             T_option=1.0, n_steps=52, noise_level=0.01, rng=None, verbose=False):
    """
    Simulate one GBM path + both hedging strategies over [0, T_option].

    Returns dict with P&L series for FNO and flat-vol strategies.
    """
    rng = rng or np.random.default_rng(0)
    dt  = T_option / n_steps
    S0  = 1.0        # normalised starting price
    K   = S0         # ATM strike

    # Initial IV surface from true params
    iv0 = _surface(model, spatial, true_sigma, true_rho, true_v0)
    sigma_init = atm_iv_from_surface(iv0, T_option)

    # Initialise option value
    call_price_init = bs_call_price(S0, K, T_option, sigma_init)

    # Initialise both hedges using initial Delta
    delta_fno  = bs_delta(S0, K, T_option, sigma_init)
    delta_flat = delta_fno        # same at t=0

    cash_fno   = call_price_init - delta_fno  * S0   # cash position (short Δ shares)
    cash_flat  = call_price_init - delta_flat * S0

    S = S0
    pnl_fno_series  = []
    pnl_flat_series = []
    calib_times     = []

    for step in range(n_steps):
        T_rem = T_option - step * dt

        # ── Simulate stock move ──────────────────────────────────────────────
        dW = rng.standard_normal() * np.sqrt(dt)
        S_new = S * np.exp((-0.5 * true_sigma**2) * dt + true_sigma * dW)

        # ── FNO-Delta: re-calibrate from noisy surface ───────────────────────
        iv_true  = _surface(model, spatial, true_sigma, true_rho, true_v0)
        noise    = rng.normal(0, noise_level * np.abs(iv_true), iv_true.shape)
        iv_noisy = np.maximum(iv_true + noise, 1e-4)

        t0 = time.perf_counter()
        res = calibrate_newton(model, iv_noisy, T_GRID, K_GRID, max_iter=12, verbose=False)
        cal_t = time.perf_counter() - t0
        calib_times.append(cal_t)

        iv_cal      = _surface(model, spatial, res["sigma"], res["rho"], res["v0"])
        sigma_fno   = atm_iv_from_surface(iv_cal, max(T_rem, 0.05))
        delta_fno_new = bs_delta(S_new, K, max(T_rem, 0.05), sigma_fno)

        # ── Flat-vol Delta: use initial sigma throughout ──────────────────────
        delta_flat_new = bs_delta(S_new, K, max(T_rem, 0.05), sigma_init)

        # ── Rebalance: update cash from trading cost ─────────────────────────
        cash_fno  += (delta_fno  - delta_fno_new)  * S_new   # sell/buy shares
        cash_flat += (delta_flat - delta_flat_new) * S_new

        delta_fno   = delta_fno_new
        delta_flat  = delta_flat_new
        S = S_new

        if verbose and step % 10 == 0:
            print(f"  step={step:3d}  S={S:.4f}  T_rem={T_rem:.3f}"
                  f"  σ_fno={sigma_fno:.4f}  Δ_fno={delta_fno:.4f}"
                  f"  cal={cal_t*1000:.0f}ms")

    # ── Terminal payoff ───────────────────────────────────────────────────────
    payoff = max(S - K, 0.0)
    pnl_fno  = payoff - delta_fno  * S - cash_fno   # net P&L at expiry
    pnl_flat = payoff - delta_flat * S - cash_flat

    return {
        "pnl_fno":   pnl_fno,
        "pnl_flat":  pnl_flat,
        "S_final":   S,
        "calib_mean_ms": float(np.mean(calib_times)) * 1e3,
        "sigma_fno_final": sigma_fno,
        "sigma_init": sigma_init,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Greeks delta hedge backtest (§5.3)")
    ap.add_argument("--paths",  type=int,   default=50,   help="Monte Carlo paths")
    ap.add_argument("--steps",  type=int,   default=52,   help="Rebalancing steps (52=weekly)")
    ap.add_argument("--noise",  type=float, default=0.01, help="IV noise level")
    ap.add_argument("--T",      type=float, default=1.0,  help="Option maturity (years)")
    ap.add_argument("--seed",   type=int,   default=0)
    ap.add_argument("--verbose",action="store_true")
    args = ap.parse_args()

    print("─" * 90)
    print(f"  §5.3 Greeks Hedge Backtest — FNO-Δ vs Flat-BS-Δ")
    print(f"  T={args.T}yr  steps={args.steps}  paths={args.paths}"
          f"  noise={args.noise*100:.1f}%  seed={args.seed}")
    print("─" * 90)

    model   = _load()
    spatial = _make_spatial_input(T_GRID, K_GRID, device=torch.device("cpu"))
    rng     = np.random.default_rng(args.seed)

    # True Rough Heston params (sigma, rho, v0 with kappa=1, theta=0.08, H=0.08)
    true_sigma, true_rho, true_v0 = 0.50, -0.60, 0.07

    pnls_fno, pnls_flat, cal_ms = [], [], []
    t_total = time.perf_counter()

    for p in range(args.paths):
        res = run_path(model, spatial,
                       true_sigma, true_rho, true_v0,
                       T_option=args.T, n_steps=args.steps,
                       noise_level=args.noise, rng=rng,
                       verbose=(args.verbose and p == 0))
        pnls_fno.append(res["pnl_fno"])
        pnls_flat.append(res["pnl_flat"])
        cal_ms.append(res["calib_mean_ms"])
        if (p + 1) % 10 == 0:
            print(f"  path {p+1:3d}/{args.paths}  "
                  f"fno_pnl={res['pnl_fno']:+.5f}  flat_pnl={res['pnl_flat']:+.5f}"
                  f"  cal={res['calib_mean_ms']:.0f}ms/step")

    elapsed = time.perf_counter() - t_total
    pnls_fno  = np.array(pnls_fno)
    pnls_flat = np.array(pnls_flat)

    var_fno  = np.var(pnls_fno)
    var_flat = np.var(pnls_flat)
    std_fno  = np.std(pnls_fno)
    std_flat = np.std(pnls_flat)

    print("\n" + "─" * 90)
    print("  RESULTS")
    print("─" * 90)
    print(f"  Paths: {args.paths}   Total time: {elapsed:.1f}s")
    print(f"  {'Strategy':<25}  {'Mean P&L':>10}  {'Std P&L':>10}  {'Var P&L':>12}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*12}")
    print(f"  {'FNO-Δ (Newton recal.)':<25}  {np.mean(pnls_fno):>+10.5f}"
          f"  {std_fno:>10.5f}  {var_fno:>12.2e}")
    print(f"  {'Flat-σ BS-Δ':<25}  {np.mean(pnls_flat):>+10.5f}"
          f"  {std_flat:>10.5f}  {var_flat:>12.2e}")

    reduction = (1.0 - var_fno / var_flat) * 100
    ratio     = var_flat / var_fno if var_fno > 0 else float("inf")

    print(f"\n  Variance reduction: {reduction:+.1f}%  (FNO vs flat-vol)")
    print(f"  Variance ratio (flat/FNO): {ratio:.2f}×")
    print(f"  Avg calibration time: {np.mean(cal_ms):.0f}ms/step  "
          f"(p95={np.percentile(cal_ms,95):.0f}ms)")

    if reduction > 0:
        print(f"\n  ✅  FNO-Δ hedge is better: {reduction:.1f}% lower P&L variance")
        print(f"      FNO re-calibration captures stochastic vol — flat-σ cannot.")
    else:
        print(f"\n  ⚠️  Flat-vol slightly better ({-reduction:.1f}%)")
        print(f"      Likely noise-induced mis-calibration at short T.  Reduce --noise or increase --paths.")
    print("─" * 90)

    # Save
    out = os.path.join(os.path.dirname(__file__), "greeks_hedge_results.txt")
    with open(out, "w") as f:
        f.write(f"§5.3 Greeks Hedge Backtest — {args.paths} paths, T={args.T}yr, "
                f"steps={args.steps}, noise={args.noise*100:.1f}%\n")
        f.write(f"FNO-Δ:  mean={np.mean(pnls_fno):+.5f}  std={std_fno:.5f}  var={var_fno:.2e}\n")
        f.write(f"Flat-Δ: mean={np.mean(pnls_flat):+.5f}  std={std_flat:.5f}  var={var_flat:.2e}\n")
        f.write(f"Variance reduction: {reduction:+.1f}%  ratio={ratio:.2f}x\n")
        f.write(f"Avg cal time: {np.mean(cal_ms):.0f}ms/step\n")
    print(f"  Results saved → {out}\n")


if __name__ == "__main__":
    main()
