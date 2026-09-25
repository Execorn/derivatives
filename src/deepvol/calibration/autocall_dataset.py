"""
autocall_dataset.py — Latin Hypercube Sampling and Batched Monte Carlo Dataset Generator.

Generates training and validation datasets for the 1-leg vanilla autocall surrogate:
  - 10-D parameter space (Heston + contract + market terms)
  - 60% ATM-dense barrier concentration
  - GPU-batched Monte Carlo pricing under Heston dynamics
  - Dataset quality control and outlier validation
"""

import os
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

from deepvol.hedging.d_xva import simulate_heston_paths, simulate_heston_paths_sobol_bb
from deepvol.models.autocall import make_obs_indices, price_autocall_mc

AUTOCALL_PARAM_BOUNDS: Dict[str, tuple] = {
    "kappa": (0.5, 5.0),
    "theta": (0.01, 0.15),
    "sigma": (0.1, 1.0),
    "rho": (-0.9, -0.1),
    "v0": (0.01, 0.15),
    "B": (0.85, 1.15),
    "coupon": (0.03, 0.25),
    "T": (0.5, 3.0),
    "n_obs_per_year": (None, None),  # discrete: {4, 8, 12}
    "r": (0.00, 0.08),
}

CORNER_BOUNDS: Dict[str, tuple] = {
    "v0": (0.01, 0.04),       # low spot variance
    "theta": (0.06, 0.15),    # high mean-reversion target (theta >> v0)
    "B": (1.03, 1.15),        # ITM barrier
    "T": (1.8, 3.0),          # long maturity
    "sigma": (0.4, 1.0),      # high vol-of-vol
    "rho": (-0.9, -0.4),      # strong leverage
    "kappa": (0.5, 5.0),      # full range
    "coupon": (0.03, 0.25),   # full range
    "r": (0.00, 0.08),        # full range
}

N_OBS_CHOICES: List[int] = [4, 8, 12]


def sample_targeted_corners(n_samples: int, seed: int = 1001) -> pd.DataFrame:
    """
    Generate parameter combinations focused on the extreme outlier corner:
    Low v0, high theta (theta >> v0), ITM barrier (B > 1.03), long T (> 1.8),
    high vol-of-vol (sigma > 0.4), and strong negative leverage (rho < -0.4).
    Uses Sobol quasi-random sequence for uniform corner space coverage.
    """
    cont_keys = ["kappa", "theta", "sigma", "rho", "v0", "B", "coupon", "T", "r"]
    l_bounds = [CORNER_BOUNDS[k][0] for k in cont_keys]
    u_bounds = [CORNER_BOUNDS[k][1] for k in cont_keys]

    sampler = qmc.Sobol(d=9, seed=seed)
    raw_samples = sampler.random(n=n_samples)
    scaled = qmc.scale(raw_samples, l_bounds, u_bounds)

    data: Dict[str, np.ndarray] = {k: scaled[:, i] for i, k in enumerate(cont_keys)}

    rng = np.random.default_rng(seed)
    u_obs = rng.uniform(0.0, 1.0, size=n_samples)
    n_obs_arr = np.where(u_obs < 0.33, 4, np.where(u_obs < 0.67, 8, 12))
    data["n_obs_per_year"] = n_obs_arr.astype(np.float64)

    columns = [
        "kappa",
        "theta",
        "sigma",
        "rho",
        "v0",
        "B",
        "coupon",
        "T",
        "n_obs_per_year",
        "r",
    ]
    return pd.DataFrame({col: data[col] for col in columns})


def sample_lhs(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate parameter combinations using Latin Hypercube Sampling with ATM-dense barrier concentration.

    Parameters:
        n_samples: Total number of rows to sample.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing 10 columns:
        kappa, theta, sigma, rho, v0, B, coupon, T, n_obs_per_year, r
    """
    cont_keys = ["kappa", "theta", "sigma", "rho", "v0", "B", "coupon", "T", "r"]
    l_bounds = [AUTOCALL_PARAM_BOUNDS[k][0] for k in cont_keys]
    u_bounds = [AUTOCALL_PARAM_BOUNDS[k][1] for k in cont_keys]

    sampler = qmc.LatinHypercube(d=9, seed=seed)
    raw_samples = sampler.random(n=n_samples)
    scaled = qmc.scale(raw_samples, l_bounds, u_bounds)

    data: Dict[str, np.ndarray] = {k: scaled[:, i] for i, k in enumerate(cont_keys)}

    # Map uniform continuous random variable to {4, 8, 12}
    rng = np.random.default_rng(seed)
    u_obs = rng.uniform(0.0, 1.0, size=n_samples)
    n_obs_arr = np.where(u_obs < 0.33, 4, np.where(u_obs < 0.67, 8, 12))
    data["n_obs_per_year"] = n_obs_arr.astype(np.float64)

    # 60% ATM-dense barrier resampling from Uniform(0.95, 1.05)
    n_atm = int(0.6 * n_samples)
    atm_indices = rng.choice(n_samples, n_atm, replace=False)
    data["B"][atm_indices] = rng.uniform(0.95, 1.05, size=n_atm)

    columns = [
        "kappa",
        "theta",
        "sigma",
        "rho",
        "v0",
        "B",
        "coupon",
        "T",
        "n_obs_per_year",
        "r",
    ]
    return pd.DataFrame({col: data[col] for col in columns})


def validate_dataset(npz_path: str) -> Dict[str, Any]:
    """
    Validate quality and physical constraints of a generated autocall dataset.
    """
    data = np.load(npz_path)
    npv = data["npv"]
    call_prob = data["call_prob"]
    exp_life = data["exp_life"]
    n_samples = len(npv)

    # Outlier count
    n_outliers = int(
        np.sum((npv < 0.5) | (npv > 1.5) | (call_prob < 0.0) | (call_prob > 1.0) | (exp_life < 0.0))
    )

    npv_stats = {
        "min": float(np.min(npv)),
        "p01": float(np.percentile(npv, 1)),
        "p50": float(np.percentile(npv, 50)),
        "p99": float(np.percentile(npv, 99)),
        "max": float(np.max(npv)),
        "mean": float(np.mean(npv)),
        "std": float(np.std(npv)),
    }

    call_stats = {
        "min": float(np.min(call_prob)),
        "p50": float(np.percentile(call_prob, 50)),
        "max": float(np.max(call_prob)),
    }

    life_stats = {
        "min": float(np.min(exp_life)),
        "p50": float(np.percentile(exp_life, 50)),
        "max": float(np.max(exp_life)),
    }

    print(f"\n--- Dataset Validation: {npz_path} ---")
    print(f"Total samples: {n_samples}, Outliers: {n_outliers}")
    print(f"NPV:       [{npv_stats['min']:.4f}, {npv_stats['max']:.4f}] | p01={npv_stats['p01']:.4f}, median={npv_stats['p50']:.4f}, p99={npv_stats['p99']:.4f}")
    print(f"Call Prob: [{call_stats['min']:.4f}, {call_stats['max']:.4f}] | median={call_stats['p50']:.4f}")
    print(f"Exp Life:  [{life_stats['min']:.4f}, {life_stats['max']:.4f}] | median={life_stats['p50']:.4f}")

    assert n_outliers == 0, f"Found {n_outliers} outliers in {npz_path}"

    return {
        "n_samples": n_samples,
        "npv_stats": npv_stats,
        "call_stats": call_stats,
        "life_stats": life_stats,
        "n_outliers": n_outliers,
    }


def generate_dataset(
    n_samples: int,
    n_paths_mc: int,
    device_str: str = "cuda",
    seed: int = 42,
    batch_size: int = 64,
    save_path: str = "data/autocall/train_100k.npz",
    use_sobol_bb: bool = False,
) -> None:
    """
    Generate parameter grid and simulate Monte Carlo autocall note prices.
    Uses sub-batched GPU execution to guarantee RTX 3060 memory safety (< 2.5 GB VRAM).

    When use_sobol_bb=True, uses Brownian Bridge observation-date Sobol anchoring
    with Milstein coarse stepping (32 steps) for ~10x faster generation with
    lower label noise (Glasserman 2004, Giles 2008).
    """
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    mode_str = "Sobol-BB + Milstein" if use_sobol_bb else "pseudo-random MC"
    print(f"Generating {n_samples} samples with {n_paths_mc} {mode_str} paths on {device}...")

    df = sample_lhs(n_samples, seed=seed)
    npv_out = np.zeros(n_samples, dtype=np.float64)
    call_prob_out = np.zeros(n_samples, dtype=np.float64)
    exp_life_out = np.zeros(n_samples, dtype=np.float64)

    # Sub-batch size: Sobol-BB uses float32 + compact obs-only storage → more headroom
    if use_sobol_bb:
        sub_batch_size = min(batch_size, 8)
    else:
        sub_batch_size = min(batch_size, 4 if n_paths_mc <= 50000 else 2)

    # Static simulation horizon across all batches to prevent CUDAGraphs recompilations
    # (only used in pseudo-random mode)
    T_sim = 3.0
    N_steps_sim = 756

    t_start = time.perf_counter()

    if use_sobol_bb:
        # Sobol-BB mode: process each sample individually (each has unique n_obs/T)
        for i in range(n_samples):
            T_i = float(df["T"].iloc[i])
            n_obs_per_yr = float(df["n_obs_per_year"].iloc[i])
            n_obs_i = max(1, int(round(n_obs_per_yr * T_i)))

            theta_np = df.iloc[i:i+1][["kappa", "theta", "sigma", "rho", "v0"]].values
            theta_t = torch.tensor(theta_np, dtype=torch.float64, device=device)
            r_i = float(df["r"].iloc[i])

            with torch.no_grad():
                S_obs, obs_idx = simulate_heston_paths_sobol_bb(
                    theta_t,
                    S0=100.0,
                    T=T_i,
                    n_obs=n_obs_i,
                    N_paths=n_paths_mc,
                    r=r_i,
                    device=device,
                    sub_steps_per_obs=8,
                    seed=i,
                    dtype=torch.float32,
                )

                # Cast observation-date values to float64 for payoff precision
                S_obs_f64 = S_obs.to(torch.float64)

                B_t = torch.tensor([df["B"].iloc[i]], dtype=torch.float64, device=device)
                coupon_t = torch.tensor([df["coupon"].iloc[i]], dtype=torch.float64, device=device)
                r_t = torch.tensor([r_i], dtype=torch.float64, device=device)

                dt_i = T_i / (n_obs_i * 8)  # Milstein sub-step dt
                npv_t, call_p_t, exp_l_t = price_autocall_mc(
                    S_obs_f64, obs_idx, B_t, coupon_t, r_t, T_i, dt_i
                )

                npv_out[i] = float(npv_t.item())
                call_prob_out[i] = float(call_p_t.item())
                exp_life_out[i] = float(exp_l_t.item())

                del S_obs, S_obs_f64
                if device.type == "cuda" and (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

            if (i + 1) % max(1, n_samples // 20) == 0 or (i + 1) == n_samples:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / max(1e-5, elapsed)
                rem = (n_samples - i - 1) / max(1e-5, rate)
                print(
                    f"[{i+1:>6d}/{n_samples}] {rate:5.1f} rows/s | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m"
                )
    else:
        # Original pseudo-random MC mode (batched with static T_sim)
        n_batches = (n_samples + sub_batch_size - 1) // sub_batch_size

        for b in range(n_batches):
            start_idx = b * sub_batch_size
            end_idx = min(start_idx + sub_batch_size, n_samples)
            curr_b_size = end_idx - start_idx

            theta_np = df.iloc[start_idx:end_idx][["kappa", "theta", "sigma", "rho", "v0"]].values
            theta_t = torch.tensor(theta_np, dtype=torch.float64, device=device)
            r_np = df.iloc[start_idx:end_idx]["r"].values
            r_batch = torch.tensor(r_np, dtype=torch.float64, device=device).unsqueeze(1)

            # Pad to full sub_batch_size if last batch to maintain static shape
            if curr_b_size < sub_batch_size:
                pad_rows = sub_batch_size - curr_b_size
                theta_pad = theta_t[-1:].repeat(pad_rows, 1)
                theta_sim = torch.cat([theta_t, theta_pad], dim=0)
                r_pad = r_batch[-1:].repeat(pad_rows, 1)
                r_sim = torch.cat([r_batch, r_pad], dim=0)
            else:
                theta_sim = theta_t
                r_sim = r_batch

            with torch.no_grad():
                S = simulate_heston_paths(
                    theta_sim,
                    S0=100.0,
                    T=T_sim,
                    N_steps=N_steps_sim,
                    N_paths=n_paths_mc,
                    r=r_sim,
                    device=device,
                )

                npv_batch = []
                call_prob_batch = []
                exp_life_batch = []

                for j in range(curr_b_size):
                    row_idx = start_idx + j
                    T_j = float(df["T"].iloc[row_idx])
                    n_obs_per_yr = float(df["n_obs_per_year"].iloc[row_idx])
                    n_obs = max(1, int(round(n_obs_per_yr * T_j)))
                    N_steps_j = max(1, int(round(T_j * 252)))
                    dt_j = T_j / N_steps_j

                    obs_indices = make_obs_indices(n_obs, T_j, N_steps_j)
                    S_j = S[j : j + 1, :, : N_steps_j + 1]

                    B_t = torch.tensor([df["B"].iloc[row_idx]], dtype=torch.float64, device=device)
                    coupon_t = torch.tensor([df["coupon"].iloc[row_idx]], dtype=torch.float64, device=device)
                    r_t = torch.tensor([df["r"].iloc[row_idx]], dtype=torch.float64, device=device)

                    npv_t, call_p_t, exp_l_t = price_autocall_mc(
                        S_j, obs_indices, B_t, coupon_t, r_t, T_j, dt_j
                    )

                    npv_batch.append(npv_t)
                    call_prob_batch.append(call_p_t)
                    exp_life_batch.append(exp_l_t)

                npv_out[start_idx:end_idx] = torch.cat(npv_batch).cpu().numpy()
                call_prob_out[start_idx:end_idx] = torch.cat(call_prob_batch).cpu().numpy()
                exp_life_out[start_idx:end_idx] = torch.cat(exp_life_batch).cpu().numpy()

                del S
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if (b + 1) % max(1, 500 // sub_batch_size) == 0 or end_idx == n_samples:
                elapsed = time.perf_counter() - t_start
                rate = end_idx / max(1e-5, elapsed)
                rem = (n_samples - end_idx) / max(1e-5, rate)
                print(
                    f"[{end_idx:>6d}/{n_samples}] {rate:5.1f} rows/s | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m"
                )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {col: df[col].values for col in df.columns}
    save_dict.update({"npv": npv_out, "call_prob": call_prob_out, "exp_life": exp_life_out})
    np.savez_compressed(save_path, **save_dict)
    print(f"Successfully saved {n_samples} samples to {save_path}")


if __name__ == "__main__":
    import os
    os.makedirs("data/autocall", exist_ok=True)
    generate_dataset(100_000, 50_000,  "cuda", 42,  64, "data/autocall/train_100k.npz")
    generate_dataset( 10_000, 100_000, "cuda", 123, 32, "data/autocall/val_10k.npz")
    validate_dataset("data/autocall/train_100k.npz")
    validate_dataset("data/autocall/val_10k.npz")
