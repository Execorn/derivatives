"""
Worst-of Multi-Asset Autocallable Dataset Generator via Latin Hypercube Sampling (LHS).

Generates parameter configurations and ground-truth GPU Monte Carlo evaluations
for training and validating the WoFAutocallEGNO surrogate.

Parameters (16 features):
  - Asset 1 Heston: kappa1, theta1, sigma1, rho_sv1, v01
  - Asset 2 Heston: kappa2, theta2, sigma2, rho_sv2, v02
  - Coupling & Contract: rho_12, B, coupon, T, n_obs_per_year, r

Outputs:
  - npv: Net present value as % of par
  - call_prob: Early redemption probability
  - exp_life: Expected lifespan in years
"""

import sys
sys.path.insert(0, "src")

import os
import time
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import qmc
import torch

from deepvol.models.autocall import make_obs_indices
from deepvol.models.wof_autocall import simulate_correlated_heston_paths, price_wof_autocall_mc

WOF_PARAM_BOUNDS = {
    "kappa1": (0.5, 5.0), "theta1": (0.01, 0.15), "sigma1": (0.1, 1.0),
    "rho_sv1": (-0.9, -0.1), "v01": (0.01, 0.15),
    "kappa2": (0.5, 5.0), "theta2": (0.01, 0.15), "sigma2": (0.1, 1.0),
    "rho_sv2": (-0.9, -0.1), "v02": (0.01, 0.15),
    "rho_12": (0.0, 0.95),
    "B": (0.85, 1.10),
    "coupon": (0.03, 0.25),
    "T": (0.5, 3.0),
    "n_obs_per_year": (None, None),
    "r": (0.00, 0.08),
}

FEATURE_NAMES = [
    "kappa1", "theta1", "sigma1", "rho_sv1", "v01",
    "kappa2", "theta2", "sigma2", "rho_sv2", "v02",
    "rho_12", "B", "coupon", "T", "n_obs_per_year", "r"
]
TARGET_NAMES = ["npv", "call_prob", "exp_life"]


def sample_wof_lhs(n_samples: int, seed: int = 42, k_contracts: int = 6) -> pd.DataFrame:
    """Sample worst-of parameter space using stratified Latin Hypercube Sampling with factorized path reuse."""
    n_market = int(np.ceil(n_samples / k_contracts))

    # 1. Market parameters LHS: 12 features
    mkt_keys = [
        "kappa1", "theta1", "sigma1", "rho_sv1", "v01",
        "kappa2", "theta2", "sigma2", "rho_sv2", "v02",
        "rho_12", "r"
    ]
    mkt_sampler = qmc.LatinHypercube(d=len(mkt_keys), seed=seed)
    mkt_raw = mkt_sampler.random(n=n_market)
    l_mkt = [WOF_PARAM_BOUNDS[k][0] for k in mkt_keys]
    u_mkt = [WOF_PARAM_BOUNDS[k][1] for k in mkt_keys]
    mkt_scaled = qmc.scale(mkt_raw, l_mkt, u_mkt)
    mkt_data = {k: mkt_scaled[:, i] for i, k in enumerate(mkt_keys)}
    df_mkt = pd.DataFrame(mkt_data)

    df_mkt_expanded = df_mkt.loc[df_mkt.index.repeat(k_contracts)].iloc[:n_samples].reset_index(drop=True)

    # 2. Contract parameters LHS: B, coupon, T, n_obs_per_year
    cont_keys = ["B", "coupon", "T"]
    d_cpn = len(cont_keys) + 1  # + n_obs_per_year
    cpn_sampler = qmc.LatinHypercube(d=d_cpn, seed=seed + 2000)
    cpn_raw = cpn_sampler.random(n=n_samples)
    l_cpn = [WOF_PARAM_BOUNDS[k][0] for k in cont_keys]
    u_cpn = [WOF_PARAM_BOUNDS[k][1] for k in cont_keys]
    cpn_scaled = qmc.scale(cpn_raw[:, :len(cont_keys)], l_cpn, u_cpn)

    cpn_data = {k: cpn_scaled[:, i] for i, k in enumerate(cont_keys)}

    obs_raw = cpn_raw[:, len(cont_keys)]
    obs_discrete = np.where(obs_raw < 0.333, 4, np.where(obs_raw < 0.667, 8, 12)).astype(float)
    cpn_data["n_obs_per_year"] = obs_discrete

    # 60% ATM concentration for barrier B in [0.90, 1.05]
    rng = np.random.default_rng(seed)
    atm_mask = rng.choice(n_samples, size=int(0.60 * n_samples), replace=False)
    cpn_data["B"][atm_mask] = rng.uniform(0.90, 1.05, size=len(atm_mask))

    df_cpn = pd.DataFrame(cpn_data)
    df = pd.concat([df_mkt_expanded, df_cpn], axis=1)[FEATURE_NAMES]
    return df


def validate_wof_dataset(npz_path: str) -> Dict[str, object]:
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
    exp_life = data["exp_life"]

    assert (call_prob >= -1e-5).all() and (call_prob <= 1.00001).all()
    assert (exp_life >= 0.0).all()

    outliers = (npv < 0.3) | (npv > 2.0) | np.isnan(npv)
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


def generate_wof_dataset(
    n_samples: int,
    n_paths_mc: int = 10000,
    device_str: str = "cuda",
    seed: int = 42,
    k_contracts: int = 6,
    batch_market: int = 8,
    save_path: str = "data/autocall/wof2_train_60k.npz",
) -> None:
    """Generate worst-of autocall dataset using GPU Monte Carlo simulation with factorized path reuse."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Generating WoF dataset: {n_samples} samples, {n_paths_mc} MC paths (k={k_contracts}) on {device}...")

    df = sample_wof_lhs(n_samples, seed=seed, k_contracts=k_contracts)
    npv_out = np.zeros(n_samples, dtype=np.float64)
    call_prob_out = np.zeros(n_samples, dtype=np.float64)
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

        theta1_t = torch.tensor(
            mkt_slice[["kappa1", "theta1", "sigma1", "rho_sv1", "v01"]].values,
            dtype=torch.float64,
            device=device,
        )
        theta2_t = torch.tensor(
            mkt_slice[["kappa2", "theta2", "sigma2", "rho_sv2", "v02"]].values,
            dtype=torch.float64,
            device=device,
        )
        rho_assets_t = torch.tensor(
            mkt_slice["rho_12"].values,
            dtype=torch.float64,
            device=device,
        )
        r_t = torch.tensor(
            mkt_slice["r"].values,
            dtype=torch.float64,
            device=device,
        )

        S1, S2 = simulate_correlated_heston_paths(
            theta1=theta1_t,
            theta2=theta2_t,
            rho_assets=rho_assets_t,
            S0_1=100.0,
            S0_2=100.0,
            T=T_sim,
            N_steps=N_steps_sim,
            N_paths=n_paths_mc,
            r=r_t,
            antithetic=True,
            device=device,
        )

        npv_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)
        cp_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)
        el_buf = torch.empty(n_rows_batch, dtype=torch.float64, device=device)

        b_t = torch.tensor(batch_df["B"].values, dtype=torch.float64, device=device)
        cpn_t = torch.tensor(batch_df["coupon"].values, dtype=torch.float64, device=device)
        r_rows_t = torch.tensor(batch_df["r"].values, dtype=torch.float64, device=device)

        buf_idx = 0
        for m_local in range(B_mkt):
            m_global = m_start + m_local
            m_row_start = m_global * k_contracts
            m_row_end = min(m_row_start + k_contracts, n_samples)
            if m_row_start >= n_samples:
                break

            S1_m = S1[m_local:m_local+1]
            S2_m = S2[m_local:m_local+1]

            for row_pos in range(m_row_start, m_row_end):
                row = df.iloc[row_pos]
                T_j = float(row["T"])
                n_obs_j = max(1, int(round(float(row["n_obs_per_year"]) * T_j)))
                N_steps_j = int(round(T_j * 252))
                obs_indices = make_obs_indices(n_obs_j, T_j, N_steps_j)

                npv_j, cp_j, el_j = price_wof_autocall_mc(
                    S1=S1_m[:, :, :N_steps_j+1],
                    S2=S2_m[:, :, :N_steps_j+1],
                    obs_indices=obs_indices,
                    B=b_t[buf_idx:buf_idx+1],
                    coupon=cpn_t[buf_idx:buf_idx+1],
                    r=r_rows_t[buf_idx:buf_idx+1],
                    T=T_j,
                    dt=T_j / N_steps_j,
                )

                npv_buf[buf_idx] = npv_j[0]
                cp_buf[buf_idx] = cp_j[0]
                el_buf[buf_idx] = el_j[0]
                buf_idx += 1

        npv_out[r_start:r_end] = npv_buf.cpu().numpy()
        call_prob_out[r_start:r_end] = cp_buf.cpu().numpy()
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
        "exp_life": exp_life_out,
    })
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved {n_samples} samples to {save_path} in {(time.perf_counter() - t_start)/60:.2f} mins.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=60000)
    parser.add_argument("--paths", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_contracts", type=int, default=6)
    parser.add_argument("--batch_market", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="data/autocall/wof2_train_60k.npz")
    args = parser.parse_args()

    generate_wof_dataset(
        n_samples=args.n,
        n_paths_mc=args.paths,
        device_str=args.device,
        seed=args.seed,
        k_contracts=args.k_contracts,
        batch_market=args.batch_market,
        save_path=args.save_path,
    )
    validate_wof_dataset(args.save_path)

