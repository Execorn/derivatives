"""
Synthetic Sequential Data Generator for HestonDynamicsLSTM.

Generates ~100,000 (sequence, label) pairs for training the LSTM that
predicts tomorrow's Heston parameters from 10 days of IV surfaces.

Pipeline:
    1. Simulate 600 correlated OU trajectories of 180 days each
       (data order: [v0, rho, sigma, theta, kappa])
    2. For each day, feed parameters into the trained HestonSurrogateMLP
       to obtain the 88-point Total Variance surface W = IV² × T
    3. Build sliding windows: (10 surfaces) → (next-day params)
    4. Split 80/10/10 on trajectories (no data leakage between splits)
    5. Save to data/seq_dataset.npz with label statistics

OU Parameters calibrated to empirical SPX dynamics (Cont & Da Fonseca, 2002; Bergomi, 2016):
    Data order: [v0, rho, sigma, theta, kappa]

Usage:
    cd path/to/derivatives
    python scripts/generate_seq_data.py [--n-traj 600] [--n-days 180]

Output:
    data/seq_dataset.npz containing:
        X_train, y_train   — (N_train, 10, 88) and (N_train, 5)
        X_val,   y_val     — (N_val,   10, 88) and (N_val,   5)
        X_test,  y_test    — (N_test,  10, 88) and (N_test,  5)
        label_mean         — (5,) training label means for Z-score norm
        label_std          — (5,) training label stds  for Z-score norm
        T_vector           — (88,) maturity grid for W ↔ IV conversion
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import joblib

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import HestonSurrogateMLP

# ── Grid constants ─────────────────────────────────────────────────────────────
MATURITIES = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])   # 8 maturities
T_VECTOR   = np.repeat(MATURITIES, 11)                                # shape (88,)

# ── OU Parameters (empirical SPX — Cont & Da Fonseca, 2002; Bergomi, 2016) ──────
# Data order: [v0, rho, sigma, theta, kappa]

OU_KAPPA = np.array([2.50, 3.50, 0.80, 1.50, 1.20])  # mean-reversion speeds
OU_MU    = np.array([0.04, -0.65, 0.40, 0.05, 3.00])  # long-run means
OU_SIGMA = np.array([0.040, 0.100, 0.150, 0.020, 0.500])  # OU diffusion

# Hard clamp bounds (empirical SPX regime, narrower than MLP training domain)
OU_LOWER = np.array([0.010, -0.90, 0.10, 0.020, 1.00])
OU_UPPER = np.array([0.150, -0.40, 0.80, 0.120, 6.00])

# Correlation matrix for the 5 Brownian motions
# Data order: [v0, rho, sigma, theta, kappa]
CORR = np.array([
    [ 1.00, -0.60,  0.30,  0.70, -0.20],  # v0
    [-0.60,  1.00, -0.10, -0.40,  0.10],  # rho
    [ 0.30, -0.10,  1.00,  0.10, -0.05],  # sigma
    [ 0.70, -0.40,  0.10,  1.00, -0.35],  # theta
    [-0.20,  0.10, -0.05, -0.35,  1.00],  # kappa
], dtype=np.float64)

# Cholesky factor for correlated noise
CHOL = np.linalg.cholesky(CORR)

# Time step: 1 trading day ≈ 1/252 year
DT = 1.0 / 252.0


# ── Feller condition repair ────────────────────────────────────────────────────

def enforce_feller(params: np.ndarray, max_iters: int = 10) -> np.ndarray:
    """
    Reduce sigma iteratively until the Feller condition 2κθ > σ² holds.
    Data order: [v0, rho, sigma, theta, kappa].
    """
    p = params.copy()
    for _ in range(max_iters):
        sigma, theta, kappa = p[2], p[3], p[4]
        if 2 * kappa * theta > sigma ** 2:
            break
        p[2] *= 0.90  # shrink sigma by 10%
    return p


# ── Single OU trajectory ───────────────────────────────────────────────────────

def simulate_ou_trajectory(
    n_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one trajectory of 5 correlated OU Heston parameters.

    Returns:
        Array of shape (n_days, 5) — raw parameter trajectory.
        Data order: [v0, rho, sigma, theta, kappa].
    """
    # Sample a realistic starting point within the OU bounds
    x = OU_MU + rng.uniform(-0.3, 0.3, size=5) * (OU_UPPER - OU_LOWER) * 0.5
    x = np.clip(x, OU_LOWER, OU_UPPER)

    traj = np.empty((n_days, 5), dtype=np.float64)

    for t in range(n_days):
        # Correlated Gaussian increments via Cholesky
        z = rng.standard_normal(5)
        dW = CHOL @ z  # correlated

        # Euler-Maruyama step: dX = κ(μ - X)dt + σ√dt dW
        drift     = OU_KAPPA * (OU_MU - x) * DT
        diffusion = OU_SIGMA * np.sqrt(DT) * dW
        x = x + drift + diffusion

        # Hard clamp to empirical bounds
        x = np.clip(x, OU_LOWER, OU_UPPER)

        # Ensure Feller condition
        x = enforce_feller(x)

        traj[t] = x

    return traj


# ── Batch surface generation via surrogate MLP ────────────────────────────────

def generate_surfaces_for_trajectory(
    params_traj: np.ndarray,
    model: HestonSurrogateMLP,
    feature_scaler,
    target_scaler,
) -> np.ndarray:
    """
    Convert a (N_days, 5) parameter trajectory to a (N_days, 88) W surface
    trajectory using the trained surrogate MLP.

    Args:
        params_traj:    Raw Heston parameters, shape (N_days, 5).
        model:          Trained HestonSurrogateMLP in eval mode.
        feature_scaler: Fitted MinMaxScaler for parameters.
        target_scaler:  Fitted StandardScaler for W surfaces.

    Returns:
        Total Variance surfaces W, shape (N_days, 88), unscaled.
    """
    params_scaled = feature_scaler.transform(params_traj.astype(np.float32))
    x_tensor      = torch.tensor(params_scaled, dtype=torch.float32)

    with torch.no_grad():
        w_scaled = model(x_tensor).numpy()

    w_surfaces = target_scaler.inverse_transform(w_scaled)
    # Clamp to physically meaningful range: W ≥ 1e-8
    w_surfaces = np.maximum(w_surfaces, 1e-8)
    return w_surfaces.astype(np.float32)


# ── Sliding window builder ─────────────────────────────────────────────────────

def build_windows(
    surfaces: np.ndarray,
    params:   np.ndarray,
    seq_len:  int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (sequence, label) pairs.

    X[i] = surfaces[i : i+seq_len]      — shape (seq_len, 88)
    y[i] = params[i+seq_len]            — shape (5,)  next-day parameters

    Args:
        surfaces: W surface trajectory, shape (N_days, 88).
        params:   Parameter trajectory,  shape (N_days, 5).
        seq_len:  Number of input days (default: 10).

    Returns:
        Tuple (X, y) with shapes (N_windows, seq_len, 88) and (N_windows, 5).
    """
    n_windows = len(surfaces) - seq_len
    X = np.stack([surfaces[i : i + seq_len] for i in range(n_windows)])
    y = params[seq_len:].astype(np.float32)
    return X, y


# ── Main generation routine ────────────────────────────────────────────────────

def generate(
    n_traj:   int = 600,
    n_days:   int = 180,
    seq_len:  int = 10,
    seed:     int = 42,
    out_path: Path = PROJECT_ROOT / "data" / "seq_dataset.npz",
) -> None:
    """
    Full pipeline: simulate OU trajectories → generate surfaces → build dataset.
    """
    rng = np.random.default_rng(seed)

    # ── Load surrogate model and scalers ──────────────────────────────────────
    print("[1/5] Loading surrogate model and scalers …")
    ARTIFACTS = PROJECT_ROOT / "artifacts"
    f_scaler  = joblib.load(ARTIFACTS / "scalers" / "feature_scaler.pkl")
    t_scaler  = joblib.load(ARTIFACTS / "scalers" / "target_scaler.pkl")

    model = HestonSurrogateMLP()
    model.load_state_dict(
        torch.load(ARTIFACTS / "weights" / "heston_best.pth", map_location="cpu",
                   weights_only=False)
    )
    model.eval()
    print(f"    Surrogate loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Simulate trajectories ─────────────────────────────────────────────────
    print(f"\n[2/5] Simulating {n_traj} OU trajectories × {n_days} days …")
    t0 = time.perf_counter()

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for i in range(n_traj):
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    Trajectory {i+1}/{n_traj}  ({elapsed:.1f}s elapsed)")

        # Simulate parameter trajectory
        params_traj = simulate_ou_trajectory(n_days, rng)

        # Generate W surfaces via surrogate
        surfaces = generate_surfaces_for_trajectory(
            params_traj, model, f_scaler, t_scaler
        )

        # Build sliding windows
        X_traj, y_traj = build_windows(surfaces, params_traj.astype(np.float32), seq_len)
        all_X.append(X_traj)
        all_y.append(y_traj)

    elapsed = time.perf_counter() - t0
    print(f"    Done — {elapsed:.1f}s total")

    # ── Concatenate all trajectories ──────────────────────────────────────────
    print("\n[3/5] Concatenating windows …")
    X_all = np.concatenate(all_X, axis=0)  # (N_total, 10, 88)
    y_all = np.concatenate(all_y, axis=0)  # (N_total, 5)
    print(f"    X_all: {X_all.shape}  y_all: {y_all.shape}")

    # ── Split on trajectory boundaries (no leakage) ───────────────────────────
    print("\n[4/5] Splitting 80/10/10 on trajectory boundaries …")
    n_train_traj = int(0.80 * n_traj)  # 480
    n_val_traj   = int(0.10 * n_traj)  # 60

    windows_per_traj = n_days - seq_len  # 170

    train_end = n_train_traj * windows_per_traj
    val_end   = train_end + n_val_traj * windows_per_traj

    X_train, y_train = X_all[:train_end],        y_all[:train_end]
    X_val,   y_val   = X_all[train_end:val_end],  y_all[train_end:val_end]
    X_test,  y_test  = X_all[val_end:],           y_all[val_end:]

    print(f"    Train : {X_train.shape[0]:>7,} sequences")
    print(f"    Val   : {X_val.shape[0]:>7,} sequences")
    print(f"    Test  : {X_test.shape[0]:>7,} sequences")

    # ── Compute label statistics from training set only ───────────────────────
    print("\n[5/5] Computing label Z-score statistics and saving …")
    label_mean = y_train.mean(axis=0)  # shape (5,)
    label_std  = y_train.std(axis=0)   # shape (5,)

    # Guard against zero std (degenerate parameter)
    label_std = np.maximum(label_std, 1e-8)

    print("    Label statistics (data order: v0, rho, sigma, theta, kappa):")
    for name, mu, sd in zip(["v0", "rho", "sigma", "theta", "kappa"], label_mean, label_std):
        print(f"      {name:6s}: mean={mu:.5f}  std={sd:.5f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        label_mean=label_mean,
        label_std=label_std,
        T_vector=T_VECTOR,
    )

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\n✓ Dataset saved → {out_path}  ({size_mb:.1f} MB)")
    print(f"  Total sequences : {len(X_all):,}")
    print(f"  Input shape     : {X_train.shape[1:]}")
    print(f"  Output shape    : {y_train.shape[1:]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic sequence dataset for HestonDynamicsLSTM"
    )
    parser.add_argument("--n-traj", type=int, default=600,
                        help="Number of OU trajectories (default: 600)")
    parser.add_argument("--n-days", type=int, default=180,
                        help="Days per trajectory (default: 180)")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Sequence window length (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    generate(
        n_traj=args.n_traj,
        n_days=args.n_days,
        seq_len=args.seq_len,
        seed=args.seed,
    )
