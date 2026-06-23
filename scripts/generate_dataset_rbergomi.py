"""
generate_dataset_rbergomi.py — Dataset generation script for Rough Bergomi.
Samples 4 parameters (v0, H, eta, rho) using Sobol sequence.
Uses adaptive steps per unit: 500 for H < 0.07, 200 for H >= 0.07.
"""

import os
import sys
import time
import numpy as np
from scipy.stats import qmc

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from src.pricing.rbergomi_gpu import batch_rbergomi_iv_surface

# ─── Config ────────────────────────────────────────────────────────────────────
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
K_GRID = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

PARAM_NAMES = ["v0", "H", "eta", "rho"]
BOUNDS_LOWER = np.array([0.01, 0.04, 0.5, -0.95], dtype=np.float32)
BOUNDS_UPPER = np.array([0.20, 0.15, 4.0, 0.0], dtype=np.float32)

N_TRAIN_SAMPLES = 50000
N_VAL_SAMPLES = 10000
BATCH_SIZE = 512

OUTPUT_PATH = os.path.join(repo_root, "data", "rBergomiDataset_v1.npz")
CHECKPOINT_PATH = os.path.join(repo_root, "data", "rBergomiDataset_checkpoint.npz")


def main():
    print("=" * 60)
    print("  Rough Bergomi Dataset Generation  ")
    print("=" * 60)
    print(f"  Train Samples : {N_TRAIN_SAMPLES:,}")
    print(f"  Val Samples   : {N_VAL_SAMPLES:,}")
    print(f"  T grid        : {T_GRID.tolist()}")
    print(f"  K grid        : {K_GRID.tolist()}")
    print(f"  Bounds        :")
    for name, low, high in zip(PARAM_NAMES, BOUNDS_LOWER, BOUNDS_UPPER):
        print(f"    {name:5s} : [{low:.4f}, {high:.4f}]")
    print(f"  Output        : {OUTPUT_PATH}")
    print()

    # Generate Sobol samples (65536 points is the next power of 2 above 60000)
    sampler = qmc.Sobol(d=4, scramble=True, seed=42)
    unit_pts = sampler.random(65536)
    params_all = qmc.scale(unit_pts, BOUNDS_LOWER, BOUNDS_UPPER).astype(np.float32)

    params_train = params_all[:N_TRAIN_SAMPLES]
    params_val = params_all[N_TRAIN_SAMPLES : N_TRAIN_SAMPLES + N_VAL_SAMPLES]

    # Containers for results
    iv_train = np.zeros((N_TRAIN_SAMPLES, len(T_GRID), len(K_GRID)), dtype=np.float32)
    iv_val = np.zeros((N_VAL_SAMPLES, len(T_GRID), len(K_GRID)), dtype=np.float32)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Recovery from checkpoint if it exists
    start_train_idx = 0
    start_val_idx = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = np.load(CHECKPOINT_PATH)
            iv_train = ckpt["iv_train"]
            iv_val = ckpt["iv_val"]
            start_train_idx = int(ckpt["train_idx"])
            start_val_idx = int(ckpt["val_idx"])
            print(f"  Resuming from checkpoint. Train index: {start_train_idx}, Val index: {start_val_idx}")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}. Starting from scratch.")

    t0 = time.time()

    # 1. Generate Training Samples (10k paths)
    if start_train_idx < N_TRAIN_SAMPLES:
        print("\n  Generating training set (10k paths)...")
        n_train_batches = (N_TRAIN_SAMPLES - start_train_idx + BATCH_SIZE - 1) // BATCH_SIZE
        for b in range(n_train_batches):
            s = start_train_idx + b * BATCH_SIZE
            e = min(s + BATCH_SIZE, N_TRAIN_SAMPLES)

            bt = time.time()
            batch_iv = batch_rbergomi_iv_surface(
                params_train[s:e],
                T_GRID,
                K_GRID,
                N_paths=10000,
                antithetic=True,
                device="cuda",
            )
            bt = time.time() - bt

            iv_train[s:e] = batch_iv

            # Checkpoint save every 1,000 surfaces or at the end
            if (e % 1000 == 0) or (e == N_TRAIN_SAMPLES):
                np.savez(
                    CHECKPOINT_PATH,
                    iv_train=iv_train,
                    iv_val=iv_val,
                    train_idx=e,
                    val_idx=0,
                )
                print(f"    Saved checkpoint at train index {e}/{N_TRAIN_SAMPLES}")

            rate = (e - start_train_idx) / (time.time() - t0 + 1e-6)
            eta_s = (N_TRAIN_SAMPLES - e + N_VAL_SAMPLES) / max(rate, 1e-6)
            print(
                f"    Batch {b+1:>5}/{n_train_batches} | Done: {e:>6,} | "
                f"Batch t: {bt:>6.2f}s | ETA: {eta_s/60:>6.1f}min"
            )

    # 2. Generate Validation Samples (50k paths)
    if start_val_idx < N_VAL_SAMPLES:
        print("\n  Generating validation set (50k paths)...")
        n_val_batches = (N_VAL_SAMPLES - start_val_idx + BATCH_SIZE - 1) // BATCH_SIZE
        t0_val = time.time()
        for b in range(n_val_batches):
            s = start_val_idx + b * BATCH_SIZE
            e = min(s + BATCH_SIZE, N_VAL_SAMPLES)

            bt = time.time()
            batch_iv = batch_rbergomi_iv_surface(
                params_val[s:e],
                T_GRID,
                K_GRID,
                N_paths=50000,
                antithetic=True,
                device="cuda",
            )
            bt = time.time() - bt

            iv_val[s:e] = batch_iv

            # Checkpoint save every 1,000 surfaces or at the end
            if (e % 1000 == 0) or (e == N_VAL_SAMPLES):
                np.savez(
                    CHECKPOINT_PATH,
                    iv_train=iv_train,
                    iv_val=iv_val,
                    train_idx=N_TRAIN_SAMPLES,
                    val_idx=e,
                )
                print(f"    Saved checkpoint at val index {e}/{N_VAL_SAMPLES}")

            rate = (e - start_val_idx) / (time.time() - t0_val + 1e-6)
            eta_s = (N_VAL_SAMPLES - e) / max(rate, 1e-6)
            print(
                f"    Batch {b+1:>5}/{n_val_batches} | Done: {e:>6,} | "
                f"Batch t: {bt:>6.2f}s | ETA: {eta_s/60:>6.1f}min"
            )

    # Save final dataset
    print(f"\n  Saving final dataset to {OUTPUT_PATH}...")
    np.savez_compressed(
        OUTPUT_PATH,
        params_train=params_train,
        iv_train=iv_train,
        params_val=params_val,
        iv_val=iv_val,
        T_grid=T_GRID,
        K_grid=K_GRID,
        param_names=np.array(PARAM_NAMES),
    )
    print("  Done!")

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
        except Exception:
            pass


if __name__ == "__main__":
    main()
