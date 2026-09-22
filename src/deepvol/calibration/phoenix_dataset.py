"""
Phoenix Autocallable Dataset Generator via Latin Hypercube Sampling (LHS).

Generates parameter configurations and ground-truth GPU Monte Carlo evaluations
for training and validating the PhoenixMLP neural surrogate.

Parameters:
  - Heston dynamics: kappa, theta, sigma, rho, v0
  - Phoenix contract terms: B_call, B_cpn (B_cpn = B_call - delta_B), coupon, T,
    n_obs_per_year, r, memory (bool)

Outputs:
  - npv: Net present value as % of par
  - call_prob: Early redemption probability
  - cpn_prob: Probability of receiving at least one coupon payment
  - exp_life: Expected lifespan in years
"""

import sys
sys.path.insert(0, "src")

import os
import time
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

from deepvol.models.autocall import make_obs_indices
from deepvol.models.phoenix import price_phoenix_mc
from deepvol.hedging.d_xva import simulate_heston_paths

PHOENIX_PARAM_BOUNDS = {
    "kappa": (0.5, 5.0),
    "theta": (0.01, 0.15),
    "sigma": (0.1, 1.0),
    "rho": (-0.9, -0.1),
    "v0": (0.01, 0.15),
    "B_call": (0.90, 1.15),
    "delta_B": (0.05, 0.30),
    "coupon": (0.03, 0.25),
    "T": (0.5, 3.0),
    "n_obs_per_year": (None, None),
    "r": (0.00, 0.08),
    "memory": (None, None),
}

N_OBS_CHOICES = [4, 8, 12]
FEATURE_NAMES = [
    "kappa", "theta", "sigma", "rho", "v0",
    "B_call", "B_cpn", "coupon", "T", "n_obs_per_year", "r", "memory"
]
TARGET_NAMES = ["npv", "call_prob", "cpn_prob", "exp_life"]


def sample_phoenix_lhs(n_samples: int, seed: int = 42, k_contracts: int = 8) -> pd.DataFrame:
    """Sample parameter space using stratified Latin Hypercube Sampling with factorized path reuse."""
    n_market = int(np.ceil(n_samples / k_contracts))

    # 1. Market parameters LHS: kappa, theta, sigma, rho, v0, r
    mkt_keys = ["kappa", "theta", "sigma", "rho", "v0", "r"]
    mkt_sampler = qmc.LatinHypercube(d=len(mkt_keys), seed=seed)
    mkt_raw = mkt_sampler.random(n=n_market)
    l_mkt = [PHOENIX_PARAM_BOUNDS[k][0] for k in mkt_keys]
    u_mkt = [PHOENIX_PARAM_BOUNDS[k][1] for k in mkt_keys]
    mkt_scaled = qmc.scale(mkt_raw, l_mkt, u_mkt)
    mkt_data = {k: mkt_scaled[:, i] for i, k in enumerate(mkt_keys)}
    df_mkt = pd.DataFrame(mkt_data)

    # Repeat market rows k_contracts times
    df_mkt_expanded = df_mkt.loc[df_mkt.index.repeat(k_contracts)].iloc[:n_samples].reset_index(drop=True)

    # 2. Contract parameters LHS: B_call, delta_B, coupon, T, n_obs, memory
    cont_contract_keys = ["B_call", "delta_B", "coupon", "T"]
    d_cpn = len(cont_contract_keys) + 2  # + n_obs + memory
    cpn_sampler = qmc.LatinHypercube(d=d_cpn, seed=seed + 1000)
    cpn_raw = cpn_sampler.random(n=n_samples)
    l_cpn = [PHOENIX_PARAM_BOUNDS[k][0] for k in cont_contract_keys]
    u_cpn = [PHOENIX_PARAM_BOUNDS[k][1] for k in cont_contract_keys]
    cpn_scaled = qmc.scale(cpn_raw[:, :len(cont_contract_keys)], l_cpn, u_cpn)

    cpn_data = {k: cpn_scaled[:, i] for i, k in enumerate(cont_contract_keys)}

    # Map discrete n_obs_per_year
    obs_raw = cpn_raw[:, len(cont_contract_keys)]
    obs_discrete = np.where(obs_raw < 0.333, 4, np.where(obs_raw < 0.667, 8, 12)).astype(float)
    cpn_data["n_obs_per_year"] = obs_discrete

    # Map discrete memory flag (0 or 1)
    mem_raw = cpn_raw[:, len(cont_contract_keys) + 1]
    cpn_data["memory"] = (mem_raw >= 0.5).astype(float)

    # 60% ATM-dense concentration for B_call and delta_B
    rng = np.random.default_rng(seed)
    atm_mask = rng.choice(n_samples, size=int(0.60 * n_samples), replace=False)
    cpn_data["B_call"][atm_mask] = rng.uniform(0.95, 1.05, size=len(atm_mask))
    cpn_data["delta_B"][atm_mask] = rng.uniform(0.05, 0.15, size=len(atm_mask))

    # Compute B_cpn = B_call - delta_B
    cpn_data["B_cpn"] = cpn_data["B_call"] - cpn_data["delta_B"]
    del cpn_data["delta_B"]

    df_cpn = pd.DataFrame(cpn_data)

    df = pd.concat([df_mkt_expanded, df_cpn], axis=1)[FEATURE_NAMES]
    return df


def validate_phoenix_dataset(npz_path: str) -> Dict[str, object]:
    """Validate generated dataset file for shape, consistency, and outliers."""
    data = np.load(npz_path)
    n_samples = len(data["npv"])
    assert n_samples > 0, "Dataset is empty!"

    for f in FEATURE_NAMES:
        assert f in data, f"Missing feature {f}"
        assert len(data[f]) == n_samples, f"Feature length mismatch: {f}"

    for t in TARGET_NAMES:
        assert t in data, f"Missing target {t}"
        assert len(data[t]) == n_samples, f"Target length mismatch: {t}"

    npv = data["npv"]
    call_prob = data["call_prob"]
    cpn_prob = data["cpn_prob"]
    exp_life = data["exp_life"]

    assert (call_prob >= -1e-5).all() and (call_prob <= 1.00001).all()
    assert (cpn_prob >= -1e-5).all() and (cpn_prob <= 1.00001).all()
    assert (exp_life >= 0.0).all()

    outliers = (npv < 0.4) | (npv > 2.0) | np.isnan(npv)
    n_outliers = int(np.sum(outliers))

    stats = {
        "n_samples": n_samples,
        "npv_min": float(np.min(npv)),
        "npv_max": float(np.max(npv)),
        "npv_mean": float(np.mean(npv)),
        "npv_p50": float(np.percentile(npv, 50)),
        "n_outliers": n_outliers,
    }
    print(f"Validated {npz_path}: {stats}")
    return stats


def generate_phoenix_dataset(
    n_samples: int,
    n_paths_mc: int = 10000,
    device_str: str = "cuda",
    seed: int = 42,
    k_contracts: int = 8,
    batch_market: int = 16,
    save_path: str = "data/autocall/phoenix_train_80k.npz",
) -> None:
    """Generate Phoenix dataset using GPU Monte Carlo simulation with factorized path reuse."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Generating Phoenix dataset: {n_samples} samples, {n_paths_mc} MC paths (k={k_contracts}) on {device}...")

    df = sample_phoenix_lhs(n_samples, seed=seed, k_contracts=k_contracts)
    npv_out = np.zeros(n_samples, dtype=np.float64)
    call_prob_out = np.zeros(n_samples, dtype=np.float64)
    cpn_prob_out = np.zeros(n_samples, dtype=np.float64)
    exp_life_out = np.zeros(n_samples, dtype=np.float64)

    T_sim = 3.0
    N_steps_sim = 756

    n_market_total = int(np.ceil(n_samples / k_contracts))
    t_start = time.perf_counter()

    for m_start in range(0, n_market_total, batch_market):
        m_end = min(m_start + batch_market, n_market_total)
        B_mkt = m_end - m_start

        r_start = m_start * k_contracts
        r_end = min(m_end * k_contracts, n_samples)
        batch_df = df.iloc[r_start:r_end]
        n_rows_batch = len(batch_df)

        mkt_indices = [min(m * k_contracts, n_samples - 1) for m in range(m_start, m_end)]
        mkt_slice = df.iloc[mkt_indices]

        theta_t = torch.tensor(
            mkt_slice[["kappa", "theta", "sigma", "rho", "v0"]].values,
            dtype=torch.float64,
            device=device,
        )
        r_t = torch.tensor(
            mkt_slice["r"].values,
            dtype=torch.float64,
            device=device,
        )

        S = simulate_heston_paths(
            theta=theta_t,
            S0=100.0,
            T=T_sim,
            N_steps=N_steps_sim,
            N_paths=n_paths_mc,
            r=r_t,
            antithetic=True,
            device=device,
        )

        npv_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)
        cp_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)
        cpn_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)
        el_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)

        b_call_t = torch.tensor(batch_df["B_call"].values, dtype=torch.float64, device=device)
        b_cpn_t = torch.tensor(batch_df["B_cpn"].values, dtype=torch.float64, device=device)
        n_obs_t = torch.tensor(batch_df["n_obs_per_year"].values, dtype=torch.float64, device=device)
        cpn_t = torch.tensor(batch_df["coupon"].values, dtype=torch.float64, device=device)
        coupon_period_t = cpn_t / n_obs_t
        r_rows_t = torch.tensor(batch_df["r"].values, dtype=torch.float64, device=device)

        buf_idx = 0
        for m_local in range(B_mkt):
            m_global = m_start + m_local
            m_row_start = m_global * k_contracts
            m_row_end = min(m_row_start + k_contracts, n_samples)
            if m_row_start >= n_samples:
                break

            S_m = S[m_local:m_local+1]

            for row_pos in range(m_row_start, m_row_end):
                row = df.iloc[row_pos]
                T_j = float(row["T"])
                n_obs_j = max(1, int(round(float(row["n_obs_per_year"]) * T_j)))
                N_steps_j = int(round(T_j * 252))
                obs_indices = make_obs_indices(n_obs_j, T_j, N_steps_j)
                mem_j = bool(row["memory"] > 0.5)

                npv_j, cp_j, cpn_j, el_j = price_phoenix_mc(
                    S=S_m[:, :, :N_steps_j+1],
                    obs_indices=obs_indices,
                    B_call=b_call_t[buf_idx:buf_idx+1],
                    B_cpn=b_cpn_t[buf_idx:buf_idx+1],
                    coupon=coupon_period_t[buf_idx:buf_idx+1],
                    r=r_rows_t[buf_idx:buf_idx+1],
                    T=T_j,
                    dt=T_j / N_steps_j,
                    memory=mem_j,
                )

                npv_buf[buf_idx] = npv_j[0]
                cp_buf[buf_idx] = cp_j[0]
                cpn_buf[buf_idx] = cpn_j[0]
                el_buf[buf_idx] = el_j[0]
                buf_idx += 1

        npv_out[r_start:r_end] = npv_buf.cpu().numpy()
        call_prob_out[r_start:r_end] = cp_buf.cpu().numpy()
        cpn_prob_out[r_start:r_end] = cpn_buf.cpu().numpy()
        exp_life_out[r_start:r_end] = el_buf.cpu().numpy()

        if r_end % 5000 < (batch_market * k_contracts) or r_end == n_samples:
            elapsed = time.perf_counter() - t_start
            rate = r_end / max(1e-5, elapsed)
            rem = (n_samples - r_end) / max(1e-5, rate)
            print(f"[{r_end:6d}/{n_samples}] {rate:.1f} rows/s | Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_dict = {col: df[col].values for col in df.columns}
    save_dict.update({
        "npv": npv_out,
        "call_prob": call_prob_out,
        "cpn_prob": cpn_prob_out,
        "exp_life": exp_life_out,
    })
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved {n_samples} samples to {save_path} in {(time.perf_counter() - t_start)/60:.2f} mins.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=80000)
    parser.add_argument("--paths", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_contracts", type=int, default=8)
    parser.add_argument("--batch_market", type=int, default=16)
    parser.add_argument("--save_path", type=str, default="data/autocall/phoenix_train_80k.npz")
    args = parser.parse_args()

    generate_phoenix_dataset(
        n_samples=args.n,
        n_paths_mc=args.paths,
        device_str=args.device,
        seed=args.seed,
        k_contracts=args.k_contracts,
        batch_market=args.batch_market,
        save_path=args.save_path,
    )
    validate_phoenix_dataset(args.save_path)

