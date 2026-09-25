"""
Active Learning Pipeline for Autocall Correction Surrogate (Phase D).

Supports:
  - Round 1: Deterministic extreme-corner injection (10,000 samples)
  - Round 2: Ensemble-guided high-uncertainty acquisition (10,000 samples from 100,000 candidates)
  - GPU-batched Monte Carlo ground truth labeling
  - CPU-parallel Gatheral second-order PDE labeling
  - Dataset merge & expansion from 100k -> 110k -> 120k
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from deepvol.calibration.autocall_dataset import sample_lhs, sample_targeted_corners
from deepvol.calibration.generate_pde_labels import compute_pde_label
from deepvol.hedging.d_xva import simulate_heston_paths_sobol_bb
from deepvol.models.autocall import price_autocall_mc
from deepvol.surrogates.correction_ensemble import CorrectionEnsemble
from deepvol.training.train_correction import (
    CorrectionInputNormalizer,
    CorrectionOutputNormalizer,
)


def run_mc_labels(
    df: pd.DataFrame,
    n_paths: int = 32768,
    device_str: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Computes Monte Carlo ground truth labels (npv, call_prob, exp_life)
    using Brownian Bridge Sobol sampling on GPU.
    """
    device = torch.device(
        device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    n_samples = len(df)
    print(f"Running GPU Monte Carlo labeling for {n_samples:,} contracts ({n_paths:,} paths) on {device}...")

    npv_out = np.zeros(n_samples, dtype=np.float64)
    call_prob_out = np.zeros(n_samples, dtype=np.float64)
    exp_life_out = np.zeros(n_samples, dtype=np.float64)

    t0 = time.perf_counter()

    for i in range(n_samples):
        T_i = float(df["T"].iloc[i])
        n_obs_per_yr = float(df["n_obs_per_year"].iloc[i])
        n_obs_i = max(1, int(round(n_obs_per_yr * T_i)))

        theta_np = df.iloc[i : i + 1][["kappa", "theta", "sigma", "rho", "v0"]].values
        theta_t = torch.tensor(theta_np, dtype=torch.float64, device=device)
        r_i = float(df["r"].iloc[i])

        with torch.no_grad():
            S_obs, obs_idx = simulate_heston_paths_sobol_bb(
                theta_t,
                S0=100.0,
                T=T_i,
                n_obs=n_obs_i,
                N_paths=n_paths,
                r=r_i,
                device=device,
                sub_steps_per_obs=8,
                seed=100000 + i,
                dtype=torch.float32,
            )

            S_obs_f64 = S_obs.to(torch.float64)
            B_t = torch.tensor([df["B"].iloc[i]], dtype=torch.float64, device=device)
            coupon_t = torch.tensor([df["coupon"].iloc[i]], dtype=torch.float64, device=device)
            r_t = torch.tensor([r_i], dtype=torch.float64, device=device)
            dt_i = T_i / (n_obs_i * 8)

            npv_t, call_p_t, exp_l_t = price_autocall_mc(
                S_obs_f64, obs_idx, B_t, coupon_t, r_t, T_i, dt_i
            )

            npv_out[i] = float(npv_t.item())
            call_prob_out[i] = float(call_p_t.item())
            exp_life_out[i] = float(exp_l_t.item())

            del S_obs, S_obs_f64
            if device.type == "cuda" and (i + 1) % 200 == 0:
                torch.cuda.empty_cache()

        if (i + 1) % max(1, n_samples // 10) == 0 or (i + 1) == n_samples:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(1e-5, elapsed)
            rem = (n_samples - i - 1) / max(1e-5, rate)
            print(
                f"  [{i+1:>6d}/{n_samples}] {rate:5.1f} samples/s | "
                f"Elapsed: {elapsed/60:.1f}m | ETA: {rem/60:.1f}m"
            )

    return {
        "npv": npv_out,
        "call_prob": call_prob_out,
        "exp_life": exp_life_out,
    }


def run_pde_labels(
    df: pd.DataFrame,
    n_workers: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Gatheral second-order PDE labels (pde_npv, pde_delta, pde_gamma)
    using CPU workers.
    """
    import tempfile
    from deepvol.calibration.generate_pde_labels import add_pde_labels

    n_samples = len(df)
    print(f"Running CPU Gatheral PDE labeling for {n_samples:,} contracts ({n_workers} workers)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "pde_in.npz")
        out_file = os.path.join(tmpdir, "pde_out.npz")

        # Save dummy targets so add_pde_labels format is satisfied
        save_dict = {col: df[col].values for col in df.columns}
        if "npv" not in save_dict:
            save_dict["npv"] = np.ones(n_samples, dtype=np.float64)
            save_dict["call_prob"] = np.zeros(n_samples, dtype=np.float64)
            save_dict["exp_life"] = np.zeros(n_samples, dtype=np.float64)

        np.savez_compressed(in_file, **save_dict)
        add_pde_labels(in_file, out_file, n_workers=n_workers)

        out_data = np.load(out_file)
        pde_npv = out_data["pde_npv"]
        pde_delta = out_data["pde_delta"]
        pde_gamma = out_data["pde_gamma"]

    return pde_npv, pde_delta, pde_gamma


def ensemble_guided_sampling(
    ensemble: CorrectionEnsemble,
    norm_in: CorrectionInputNormalizer,
    n_candidates: int = 100000,
    n_select: int = 10000,
    seed: int = 2026,
    device_str: str = "cuda",
) -> pd.DataFrame:
    """
    Evaluates epistemic uncertainty across a candidate pool and selects the top n_select
    highest-uncertainty contracts.
    """
    device = torch.device(
        device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Generating {n_candidates:,} candidate contracts via LHS (seed={seed})...")
    df_candidates = sample_lhs(n_candidates, seed=seed)

    # Convert candidates to dict format for build_feature_matrix
    # Pre-calculate true Gatheral PDE labels to evaluate on valid 19D manifold
    pde_npv, pde_delta, pde_gamma = run_pde_labels(df_candidates, n_workers=min(os.cpu_count() or 4, 12))
    cand_dict["pde_npv"] = pde_npv
    cand_dict["pde_delta"] = pde_delta
    cand_dict["pde_gamma"] = pde_gamma

    X_candidates = CorrectionInputNormalizer.build_feature_matrix(cand_dict)
    X_norm = norm_in.transform(X_candidates)

    print(f"Scoring {n_candidates:,} candidates with Deep Ensemble uncertainty on {device}...")
    ensemble.to(device)
    ensemble.eval()

    batch_size = 8192
    uncertainties = np.zeros(n_candidates, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_candidates, batch_size):
            end = min(n_candidates, start + batch_size)
            x_batch = torch.tensor(X_norm[start:end], dtype=torch.float32, device=device)
            _, std_batch = ensemble(x_batch)
            uncertainties[start:end] = std_batch.squeeze(-1).cpu().numpy()

    # Select top n_select highest-uncertainty indices
    top_indices = np.argsort(uncertainties)[-n_select:]
    top_indices = np.sort(top_indices)

    print(
        f"Selected top {n_select:,} candidates: "
        f"Uncertainty min={uncertainties[top_indices].min():.4f}, "
        f"median={np.median(uncertainties[top_indices]):.4f}, "
        f"max={uncertainties[top_indices].max():.4f}"
    )

    return df_candidates.iloc[top_indices].reset_index(drop=True)


def append_to_dataset(
    existing_npz_path: str,
    new_data: Dict[str, np.ndarray],
    output_npz_path: str,
) -> None:
    """Concatenates new samples to existing dataset and saves to output_npz_path."""
    existing = np.load(existing_npz_path)
    merged: Dict[str, np.ndarray] = {}

    for k in existing.files:
        if k in new_data:
            merged[k] = np.concatenate([existing[k], new_data[k]], axis=0)
        else:
            merged[k] = existing[k]

    # Verify all arrays have identical length
    lengths = {k: len(v) for k, v in merged.items()}
    assert len(set(lengths.values())) == 1, f"Mismatched array lengths: {lengths}"

    total_samples = list(lengths.values())[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_npz_path)), exist_ok=True)
    np.savez_compressed(output_npz_path, **merged)
    print(f"Appended {len(new_data['npv']):,} new samples. Saved {total_samples:,} total samples to {output_npz_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Active Learning Round for Autocall Surrogate")
    parser.add_argument("--round", type=int, choices=[1, 2], required=True, help="Active learning round (1=corner, 2=ensemble-guided)")
    parser.add_argument("--input", type=str, default="data/autocall/train_100k_sobol_pde.npz", help="Existing input dataset")
    parser.add_argument("--output", type=str, required=True, help="Output expanded dataset path")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of new samples to acquire")
    parser.add_argument("--n_paths", type=int, default=32768, help="Monte Carlo simulation paths")
    parser.add_argument("--workers", type=int, default=12, help="CPU workers for PDE labeling")
    parser.add_argument("--weights_dir", type=str, default="artifacts/weights", help="Directory with ensemble weights")
    parser.add_argument("--norm_in_path", type=str, default="artifacts/scalers/correction_input_normalizer.npz")
    args = parser.parse_args()

    if args.round == 1:
        print(f"\n=======================================================")
        print(f"ACTIVE LEARNING ROUND 1: Targeted Corner Acquisition ({args.n_samples:,} samples)")
        print(f"=======================================================\n")
        df_new = sample_targeted_corners(args.n_samples, seed=1001)
    else:
        print(f"\n=======================================================")
        print(f"ACTIVE LEARNING ROUND 2: Ensemble-Guided Uncertainty Acquisition ({args.n_samples:,} samples)")
        print(f"=======================================================\n")
        norm_in = CorrectionInputNormalizer.load(args.norm_in_path)
        ensemble = CorrectionEnsemble(K=5, in_dim=19, hidden=256, n_layers=4)
        member_paths = [os.path.join(args.weights_dir, f"autocall_correction_mlp_member_{k}.pth") for k in range(5)]
        ensemble.load_members(member_paths, device="cuda" if torch.cuda.is_available() else "cpu")
        df_new = ensemble_guided_sampling(ensemble, norm_in, n_candidates=100000, n_select=args.n_samples, seed=2026)

    # Compute GPU Monte Carlo ground truth
    mc_labels = run_mc_labels(df_new, n_paths=args.n_paths)

    # Compute CPU Gatheral PDE ground truth
    pde_npv, pde_delta, pde_gamma = run_pde_labels(df_new, n_workers=args.workers)
    residual_npv = mc_labels["npv"] - pde_npv

    # Package new sample dictionary
    new_data = {col: df_new[col].values for col in df_new.columns}
    new_data.update(mc_labels)
    new_data["pde_npv"] = pde_npv
    new_data["pde_delta"] = pde_delta
    new_data["pde_gamma"] = pde_gamma
    new_data["residual_npv"] = residual_npv

    # Append to existing dataset
    append_to_dataset(args.input, new_data, args.output)
    print(f"\nRound {args.round} Active Learning completed successfully!")


if __name__ == "__main__":
    main()

