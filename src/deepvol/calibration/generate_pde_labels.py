"""PDE label generator for multi-fidelity autocall correction (Phase B).

Adds PDE NPV, Delta, Gamma labels from a 1D Crank-Nicolson BS-PDE solver
(with flat effective vol sigma=sqrt(v0)) to existing Sobol-BB MC datasets.

Maximally parallelized: uses ALL available CPU cores via ProcessPoolExecutor.
The tridiagonal Thomas solver in SciPy is inherently serial in N_S but each
sample is independent → embarrassingly parallel across samples.

Performance: ~12 samples/s per core → 100k in ~70s on 12 cores.

References:
    - Gatheral (2006) 'The Volatility Surface', ch. 3: effective local vol
    - Rannacher (1984): smoothing for Crank-Nicolson oscillations

Usage:
    python -m deepvol.calibration.generate_pde_labels \\
        --input data/autocall/train_100k_sobol.npz \\
        --output data/autocall/train_100k_sobol_pde.npz
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from typing import Tuple
import numpy as np

from deepvol.models.autocall_pde import price_autocall_pde_scalar


def compute_pde_label(
    v0: float, B: float, coupon: float, T: float,
    n_obs_per_year: float, r: float,
    kappa: float = 2.0, theta: float = 0.04, sigma: float = 0.3,
) -> Tuple[float, float, float]:
    """Compute 1D BS-PDE price with Gatheral second-order effective vol.

    Uses Heston variance swap approximation:
      σ_eff²(t) = v0·e^{-κt} + θ·(1-e^{-κt}) + σ²/(4κ)·(1-e^{-κt})

    This captures the term structure from mean reversion and vol-of-vol
    convexity, but not skew (ρ effect). The MLP correction learns the
    residual from ρ-induced skew and higher-order effects.

    Ref: Gatheral (2006), 'The Volatility Surface', ch. 3, eq. (3.6).

    Returns:
        (npv, delta, gamma) — PDE price and spatial Greeks at S0=100.
    """
    def sigma_func(t: float, S: np.ndarray) -> np.ndarray:
        ekt = np.exp(-kappa * max(float(t), 1e-10))
        var_t = v0 * ekt + theta * (1.0 - ekt) + (sigma**2 / (4.0 * max(kappa, 1e-4))) * (1.0 - ekt)
        sig_val = float(np.clip(np.sqrt(max(var_t, 1e-8)), 0.01, 2.0))
        return np.full_like(S, sig_val)

    n_obs = int(round(T * n_obs_per_year))
    N_T = max(252, n_obs * 16)
    obs_indices = [int(round(i * N_T / n_obs)) for i in range(1, n_obs + 1)]
    result = price_autocall_pde_scalar(
        S0_val=100.0, r=float(r), T=float(T), N_S=300, N_T=N_T,
        obs_indices=obs_indices, B=float(B), coupon=float(coupon),
        sigma_func=sigma_func,
        rannacher_steps=2,
    )
    return result["npv"], result["delta"], result["gamma"]


_WORKER_CODE = '''import sys
import os
import numpy as np
from scipy.linalg import solve_banded

shard_id = int(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
input_path = sys.argv[4]
output_file = sys.argv[5]
prog_file = sys.argv[6]

data = np.load(input_path)
count = end - start
npv_arr = np.zeros(count, dtype=np.float64)
delta_arr = np.zeros(count, dtype=np.float64)
gamma_arr = np.zeros(count, dtype=np.float64)

v0_all = data["v0"][start:end].astype(np.float64)
B_all = data["B"][start:end].astype(np.float64)
coupon_all = data["coupon"][start:end].astype(np.float64)
T_all = data["T"][start:end].astype(np.float64)
n_obs_all = data["n_obs_per_year"][start:end].astype(np.float64)
r_all = data["r"][start:end].astype(np.float64)
kappa_all = data["kappa"][start:end].astype(np.float64) if "kappa" in data else np.full(count, 2.0, dtype=np.float64)
theta_all = data["theta"][start:end].astype(np.float64) if "theta" in data else np.full(count, 0.04, dtype=np.float64)
sigma_all = data["sigma"][start:end].astype(np.float64) if "sigma" in data else np.full(count, 0.3, dtype=np.float64)

report_interval = max(50, count // 20)

for i in range(count):
    v0 = float(v0_all[i])
    B = float(B_all[i])
    coupon = float(coupon_all[i])
    T = float(T_all[i])
    n_obs_per_year = float(n_obs_all[i])
    r = float(r_all[i])
    kappa = float(kappa_all[i])
    theta = float(theta_all[i])
    sigma = float(sigma_all[i])

    sigma_atm = float(np.clip(np.sqrt(max(v0, 1e-8)), 0.01, 2.0))
    n_obs = int(round(T * n_obs_per_year))
    N_T = max(252, n_obs * 16)
    N_S = 300
    dt = T / N_T
    S0 = 100.0
    rannacher_steps = 2

    obs_indices = [int(round(k * N_T / n_obs)) for k in range(1, n_obs + 1)]
    obs_set = set(obs_indices)
    barrier_level = B * S0

    S_min = max(1e-3, S0 * np.exp(-6.0 * sigma_atm * np.sqrt(T)))
    S_max = S0 * np.exp(6.0 * sigma_atm * np.sqrt(T))
    S_grid = np.exp(np.linspace(np.log(S_min), np.log(S_max), N_S))
    t_grid = np.linspace(0.0, T, N_T + 1)

    h_minus = S_grid[1:-1] - S_grid[:-2]
    h_plus = S_grid[2:] - S_grid[1:-1]
    h_sum = h_minus + h_plus
    S_mid = S_grid[1:-1]
    S_mid_sq = S_mid ** 2

    # Precalculate time-independent spatial geometry terms
    diff_geom_a = (2.0 * 0.5 * S_mid_sq) / (h_minus * h_sum)
    diff_geom_c = (2.0 * 0.5 * S_mid_sq) / (h_plus * h_sum)
    diff_geom_b = -(2.0 * 0.5 * S_mid_sq) / (h_plus * h_minus)
    drift_c = r * S_mid
    drift_a = -drift_c / h_sum
    drift_c_term = drift_c / h_sum

    V = np.ones(N_S, dtype=np.float64)
    if N_T in obs_set:
        V[S_grid >= barrier_level] = 1.0 + coupon * T

    obs_times = sorted([float(oi) * dt for oi in obs_indices])

    be_remaining = 0
    for k in range(N_T - 1, -1, -1):
        t_k = float(t_grid[k])
        ekt = np.exp(-kappa * max(t_k, 1e-10))
        var_k = v0 * ekt + theta * (1.0 - ekt) + (sigma**2 / (4.0 * max(kappa, 1e-4))) * (1.0 - ekt)
        sig_k = float(np.clip(np.sqrt(max(var_k, 1e-8)), 0.01, 2.0))
        sig2_k = sig_k ** 2

        a_op = sig2_k * diff_geom_a + drift_a
        c_op = sig2_k * diff_geom_c + drift_c_term
        b_op = sig2_k * diff_geom_b - r

        V0k = float(np.exp(-r * (T - t_k)))
        future = [t for t in obs_times if t >= t_k - 1e-8]
        if future:
            tn = future[0]
            Vek = float(np.exp(-r * (tn - t_k)) * (1.0 + coupon * tn))
        else:
            Vek = float(np.exp(-r * (T - t_k)))

        tw = 1.0 if be_remaining > 0 else 0.5
        ew = 1.0 - tw
        As = -tw * dt * a_op
        Ad = 1.0 - tw * dt * b_op
        Au = -tw * dt * c_op

        rhs = (ew * dt * a_op) * V[:-2] + (1.0 + ew * dt * b_op) * V[1:-1] + (ew * dt * c_op) * V[2:]
        rhs[0] -= As[0] * V0k
        rhs[-1] -= Au[-1] * Vek

        ab = np.zeros((3, N_S - 2), dtype=np.float64)
        ab[0, 1:] = Au[:-1]; ab[1, :] = Ad; ab[2, :-1] = As[1:]
        Vn = np.empty_like(V)
        Vn[0] = V0k; Vn[-1] = Vek; Vn[1:-1] = solve_banded((1, 1), ab, rhs)
        V = Vn

        if be_remaining > 0:
            be_remaining -= 1
        if k > 0 and k in obs_set:
            V[S_grid >= barrier_level] = 1.0 + coupon * t_k
            if rannacher_steps > 0:
                be_remaining = rannacher_steps

    hm = S_grid[1:-1] - S_grid[:-2]
    hp = S_grid[2:] - S_grid[1:-1]
    hs = hm + hp
    di = (V[2:] - V[:-2]) / hs
    d0 = (V[1] - V[0]) / (S_grid[1] - S_grid[0])
    de = (V[-1] - V[-2]) / (S_grid[-1] - S_grid[-2])
    dg = np.concatenate([[d0], di, [de]])
    gi = 2.0 * ((V[2:] - V[1:-1]) / hp - (V[1:-1] - V[:-2]) / hm) / hs
    gg = np.concatenate([[0.0], gi, [0.0]])

    npv_arr[i] = float(np.interp(S0, S_grid, V))
    delta_arr[i] = float(np.interp(S0, S_grid, dg))
    gamma_arr[i] = float(np.interp(S0, S_grid, gg))

    if (i + 1) % report_interval == 0 or (i + 1) == count:
        try:
            with open(prog_file, "w") as pf:
                pf.write(str(i + 1))
        except OSError:
            pass

np.savez_compressed(output_file, start=start, end=end, npv=npv_arr, delta=delta_arr, gamma=gamma_arr)
'''


def add_pde_labels(
    input_path: str,
    output_path: str,
    n_workers: int = 0,
) -> None:
    """Add PDE labels to existing MC dataset.

    Reads input_path .npz, runs CPU PDE solver in parallel for all samples,
    and saves augmented dataset to output_path with 4 new columns:
    pde_npv, pde_delta, pde_gamma, residual_npv.

    Uses independent subprocess shards to guarantee true multi-core
    parallelism without PyTorch or OpenBLAS lock interference.

    Args:
        input_path: Path to existing .npz with MC labels
        output_path: Path to save augmented .npz
        n_workers: Number of CPU parallel workers (0 = use all cores)
    """
    data = np.load(input_path)
    n = len(data["npv"])

    if n_workers <= 0:
        n_workers = max(1, os.cpu_count() or 4)

    n_workers = min(n_workers, n)
    print(f"Generating PDE labels for {n:,} samples using {n_workers} CPU workers...")

    with tempfile.TemporaryDirectory() as tmpdir:
        worker_script = os.path.join(tmpdir, "_worker.py")
        with open(worker_script, "w") as f:
            f.write(_WORKER_CODE)

        shard_size = (n + n_workers - 1) // n_workers
        procs = []
        shard_outputs = []
        shard_progs = []

        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""

        t0 = time.perf_counter()

        for s in range(n_workers):
            start = s * shard_size
            end = min(n, (s + 1) * shard_size)
            if start >= end:
                continue

            out_file = os.path.join(tmpdir, f"shard_{s}.npz")
            prog_file = os.path.join(tmpdir, f"shard_{s}.prog")
            shard_outputs.append((start, end, out_file))
            shard_progs.append(prog_file)

            cmd = [
                sys.executable,
                worker_script,
                str(s),
                str(start),
                str(end),
                os.path.abspath(input_path),
                out_file,
                prog_file,
            ]
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

        last_print = 0.0
        while any(p.poll() is None for p in procs):
            time.sleep(0.5)
            now = time.perf_counter()
            if now - last_print >= 2.0:
                completed = 0
                for pf in shard_progs:
                    if os.path.exists(pf):
                        try:
                            with open(pf, "r") as f:
                                completed += int(f.read().strip())
                        except (ValueError, OSError):
                            pass
                elapsed = max(0.1, now - t0)
                rate = completed / elapsed
                eta = (n - completed) / rate if rate > 0 else 0
                print(f"  [{completed:>7,}/{n:,}]  {rate:.0f} samples/s  "
                      f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)
                last_print = now

        for p in procs:
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"PDE worker process failed with exit code {ret}")

        elapsed = time.perf_counter() - t0
        print(f"PDE label generation done in {elapsed:.1f}s ({n / max(0.1, elapsed):.0f} samples/s)")

        pde_npv = np.zeros(n, dtype=np.float64)
        pde_delta = np.zeros(n, dtype=np.float64)
        pde_gamma = np.zeros(n, dtype=np.float64)

        for start, end, out_file in shard_outputs:
            shard_data = np.load(out_file)
            pde_npv[start:end] = shard_data["npv"]
            pde_delta[start:end] = shard_data["delta"]
            pde_gamma[start:end] = shard_data["gamma"]

    mc_npv = data["npv"].astype(np.float64)
    residual_npv = mc_npv - pde_npv

    print(f"  Residual stats: mean={residual_npv.mean():.6f}, "
          f"std={residual_npv.std():.6f}, "
          f"min={residual_npv.min():.6f}, max={residual_npv.max():.6f}")
    ratio = np.abs(residual_npv).mean() / max(np.abs(mc_npv).mean(), 1e-10)
    print(f"  PDE base |NPV| mean={np.abs(pde_npv).mean():.6f}, "
          f"|residual|/|NPV| = {ratio:.4f}")

    merged = {k: data[k] for k in data.files}
    merged["pde_npv"] = pde_npv
    merged["pde_delta"] = pde_delta
    merged["pde_gamma"] = pde_gamma
    merged["residual_npv"] = residual_npv

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **merged)
    print(f"Saved augmented dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add PDE labels to autocall dataset")
    parser.add_argument("--input", type=str, required=True, help="Input .npz path")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers (0 = use all cores)")
    args = parser.parse_args()
    add_pde_labels(args.input, args.output, args.workers)
