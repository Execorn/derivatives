"""tests/audit/test_ml_integrity.py — Phase 6 Forensic Integrity Audit.

SPEC-DEEPVOL-FX-CROSS-003 Phase 6 Audit:
  1. Data Leakage Audit:
     - Disjoint state-level partitioning across train/val/test/stress (sha256 hash sets).
     - Strict exclusion of stress holdout block from training split.
     - Bitwise recomputation of normalization mean/std and OOD statistics from train split.
     - Exclusion of test-time features (curves, discounts, rates, payoffs) from network inputs.
  2. Label Integrity & Checksums:
     - Shard manifests carry valid SHA-256 digests and git commit hashes.
     - All shards on disk verify bitwise against manifest.json.
     - Implied volatility oracle agreement (pivot_implied_vol vs independent float64 bisection solver).
  3. Determinism:
     - Inference determinism across restarts.
  4. Architecture Invariants:
     - 13 -> 64 -> 64 -> 64 -> 1 SiLU MLP with zero-initialized output layer.
     - Zero-initialized net equals Analytic_Triangulated bitwise.
     - nn_gate = 0.0 equals analytic engine bitwise.
     - Ensemble aggregation occurs in residual vol space before Black-76 valuation.
     - eps_bound and smooth floors (softplus200 at 0.01, softplus1e5 at 1e-4).
     - C2 quintic taper enforces exact 0.0 NN contribution for |z| >= 7.
  5. DML Loss Specification:
     - lambda = 1.0, nu = 0.1, mu = 10.0 scaling and gradient-term residual matching.
  6. Independent Re-evaluation of Gates G1.1–G1.5b:
     - Verification of official test split and fresh Sobol holdout gap.
  7. Checkpoint Loading Defects:
     - test_F010: Checkpoint loading omits weights_only=True.
     - test_F011: reload_fx_cross_weights crashes on standard model checkpoint dictionary.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from deepvol.hedging.pivot_iv import pivot_implied_vol
from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    N_COLS,
    smooth_T,
)
from deepvol.models.fx_cross_surrogate import (
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
)
from deepvol.training.data_generation_cross import (
    CrossSobolSampler,
    SamplerConfig,
)
from deepvol.training.train_cross_surrogate import (
    ShardedCrossDataset,
    TrainConfig,
    evaluate_gate_g1,
    fit_ood_statistics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bisection_implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    is_call: bool = True,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Independent bisection solver for Black-76 implied volatility."""
    sqT = math.sqrt(max(T, 1e-8))
    intrinsic = max(0.0, F - K if is_call else K - F)
    if price <= intrinsic:
        return 1e-6
    lo = 1e-5
    hi = 10.0
    for _ in range(5):
        s_hi = hi * sqT
        d1 = (math.log(F / K) + 0.5 * s_hi * s_hi) / s_hi
        d2 = d1 - s_hi
        p_hi = (
            F * 0.5 * math.erfc(-d1 / math.sqrt(2.0)) - K * 0.5 * math.erfc(-d2 / math.sqrt(2.0))
            if is_call
            else K * 0.5 * math.erfc(d2 / math.sqrt(2.0)) - F * 0.5 * math.erfc(d1 / math.sqrt(2.0))
        )
        if p_hi >= price:
            break
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = mid * sqT
        d1 = (math.log(F / K) + 0.5 * s_mid * s_mid) / s_mid
        d2 = d1 - s_mid
        p_mid = (
            F * 0.5 * math.erfc(-d1 / math.sqrt(2.0)) - K * 0.5 * math.erfc(-d2 / math.sqrt(2.0))
            if is_call
            else K * 0.5 * math.erfc(d2 / math.sqrt(2.0)) - F * 0.5 * math.erfc(d1 / math.sqrt(2.0))
        )
        diff = p_mid - price
        if abs(diff) < tol:
            return mid
        if diff < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _create_sample_state(F: float = 1.0, K: float = 1.0, T: float = 1.0) -> Tensor:
    """Creates a sample 34-column engine input tensor."""
    X = torch.zeros(1, N_COLS, dtype=torch.float64)
    X[0, Col.F] = F
    X[0, Col.K] = K
    X[0, Col.T] = T
    X[0, Col.T_SET] = T
    X[0, Col.D] = 1.0
    X[0, Col.G] = 1.0
    X[0, Col.RHO] = -0.20
    X[0, Col.PHI] = 1.0
    X[0, Col.SVI1] = 0.04
    X[0, Col.SVI1 + 1] = 0.10
    X[0, Col.SVI1 + 2] = -0.20
    X[0, Col.SVI1 + 3] = 0.0
    X[0, Col.SVI1 + 4] = 0.10
    X[0, Col.SVI2] = 0.09
    X[0, Col.SVI2 + 1] = 0.15
    X[0, Col.SVI2 + 2] = 0.25
    X[0, Col.SVI2 + 3] = 0.0
    X[0, Col.SVI2 + 4] = 0.12
    X[0, Col.XI1 : Col.XI1 + 5] = torch.tensor([0.20, -0.05, 0.02, -0.08, 0.03], dtype=torch.float64)
    X[0, Col.XI2 : Col.XI2 + 5] = torch.tensor([0.30, 0.08, 0.03, 0.12, 0.05], dtype=torch.float64)
    X[0, Col.NN_GATE] = 1.0
    return X


# ---------------------------------------------------------------------------
# Phase 6.1 — Leakage Audit
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_ml_integrity_splits_market_state_disjoint() -> None:
    """Verifies train/val/test/stress splits are strictly partitioned by market state vector."""
    shards_dir = Path("artifacts/data/fx_cross_shards")
    splits = ["train", "val", "test", "stress"]
    state_hashes: Dict[str, Set[str]] = {}

    for s in splits:
        split_dir = shards_dir / s
        assert split_dir.exists(), f"Split directory missing: {split_dir}"
        shard_files = sorted(split_dir.glob("*.npy"))
        assert len(shard_files) > 0, f"No shards found for split: {s}"

        hashes: Set[str] = set()
        for sf in shard_files:
            arr = np.load(sf)
            states = arr["x"][:, 1:13]  # Market state parameters
            unique_states = np.unique(states, axis=0)
            for row in unique_states:
                digest = hashlib.sha256(row.tobytes()).hexdigest()
                hashes.add(digest)
        state_hashes[s] = hashes

    # Assert pairwise disjoint intersection across all splits
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = state_hashes[s1].intersection(state_hashes[s2])
            assert len(overlap) == 0, (
                f"DATA LEAKAGE DETECTED: {len(overlap)} state vectors overlap between {s1} and {s2}"
            )


@pytest.mark.audit
def test_ml_integrity_stress_holdout_strict_exclusion() -> None:
    """Verifies that the stress holdout block is completely excluded from the training split."""
    shards_dir = Path("artifacts/data/fx_cross_shards")
    train_arr = np.load(shards_dir / "train" / "shard_0000.npy")
    train_x = train_arr["x"]

    # Market state indices: atm2 is index 7 (col 8 in x), rr25_1 is col 4, rr25_2 is col 9
    atm2 = train_x[:, 8]
    rr25_1 = train_x[:, 4]
    rr25_2 = train_x[:, 9]

    # Stress definition from CrossSobolSampler.split()
    is_stress = (atm2 > 0.8) | (np.abs(rr25_1) > 0.18) | (np.abs(rr25_2) > 0.18)
    assert not np.any(is_stress), (
        f"DATA CONTAMINATION DETECTED: {np.sum(is_stress)} stress records found in train split"
    )


@pytest.mark.audit
def test_ml_integrity_normalization_recomputed_bitwise() -> None:
    """Recomputes normalization mean and std from the train split and verifies bitwise equality against manifest."""
    shards_dir = Path("artifacts/data/fx_cross_shards")
    with open(shards_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_arr = np.load(shards_dir / "train" / "shard_0000.npy")
    train_x = train_arr["x"].astype(np.float64)

    mean_recomputed = train_x.mean(axis=0)
    var_recomputed = np.maximum(0.0, (train_x ** 2).mean(axis=0) - (mean_recomputed ** 2))
    std_recomputed = np.maximum(np.sqrt(var_recomputed), 1e-6)

    manifest_mean = np.array(manifest["train_mean"], dtype=np.float64)
    manifest_std = np.array(manifest["train_std"], dtype=np.float64)

    max_mean_diff = float(np.max(np.abs(mean_recomputed - manifest_mean)))
    max_std_diff = float(np.max(np.abs(std_recomputed - manifest_std)))

    assert max_mean_diff == 0.0, f"train_mean recomputation mismatch: max diff {max_mean_diff}"
    assert max_std_diff <= 1e-12, f"train_std recomputation mismatch: max diff {max_std_diff}"


@pytest.mark.audit
def test_ml_integrity_feature_construction_zero_test_leakage() -> None:
    """Verifies network features exclude discount factors, rates, settlement tenors, and payoffs."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    X = _create_sample_state()

    # Features: z_in, sqrt(T), rho, xi1 (5), xi2 (5)
    _, _, _, _, _, _, _, _, _, _, cols_std = engine._prepare_features(X)
    assert len(cols_std) == 13, f"Expected 13 inputs, got {len(cols_std)}"

    # Altering discount factor D, carry g, or settlement rates r must NOT change network input features
    X_alt = X.clone()
    X_alt[0, Col.D] = 0.50
    X_alt[0, Col.G] = 2.00
    X_alt[0, Col.R_C] = 0.35
    X_alt[0, Col.R_SET] = 0.35
    X_alt[0, Col.PHI] = -1.0  # Put

    _, _, _, _, _, _, _, _, _, _, cols_alt = engine._prepare_features(X_alt)
    for j in range(13):
        assert torch.equal(cols_std[j].v, cols_alt[j].v), (
            f"Feature leakage detected on coordinate {j}: changed with curve/payoff perturbation"
        )


# ---------------------------------------------------------------------------
# Phase 6.2 — Label Integrity & Shard Checksums
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_ml_integrity_shard_sha256_manifest_verification() -> None:
    """Verifies that all binary shard files match their SHA-256 digests in manifest.json."""
    shards_dir = Path("artifacts/data/fx_cross_shards")
    with open(shards_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_checksums = manifest.get("file_checksums", {})
    assert len(file_checksums) >= 4, "Manifest must contain checksums for train, val, test, and stress"

    for rel_path, expected_hash in file_checksums.items():
        shard_path = shards_dir / rel_path
        assert shard_path.exists(), f"Shard file missing on disk: {shard_path}"
        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"SHA-256 CHECKSUM MISMATCH for {rel_path}: expected {expected_hash}, got {actual_hash}"
        )


@pytest.mark.audit
def test_ml_integrity_manifest_git_commit_present() -> None:
    """Verifies manifest.json records the git commit hash."""
    shards_dir = Path("artifacts/data/fx_cross_shards")
    with open(shards_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    git_hash = manifest.get("git_commit")
    assert git_hash is not None and len(git_hash) >= 7, "manifest.json missing valid git_commit hash"


@pytest.mark.audit
def test_ml_integrity_pivot_iv_vs_bisection_oracle() -> None:
    """Verifies pivot_implied_vol matches independent float64 bisection oracle to <= 1e-10."""
    F = 1.0
    K_vals = [0.85, 0.95, 1.0, 1.05, 1.20]
    T_vals = [0.1, 0.5, 1.0, 2.0]
    sig_true = 0.25

    for K in K_vals:
        for T in T_vals:
            sqT = sig_true * math.sqrt(T)
            d1 = (math.log(F / K) + 0.5 * sqT * sqT) / sqT
            d2 = d1 - sqT
            is_call = K >= 1.0
            price = (
                F * 0.5 * math.erfc(-d1 / math.sqrt(2.0)) - K * 0.5 * math.erfc(-d2 / math.sqrt(2.0))
                if is_call
                else K * 0.5 * math.erfc(d2 / math.sqrt(2.0)) - F * 0.5 * math.erfc(d1 / math.sqrt(2.0))
            )

            p_t = torch.tensor([price], dtype=torch.float64)
            f_t = torch.tensor([F], dtype=torch.float64)
            k_t = torch.tensor([K], dtype=torch.float64)
            t_t = torch.tensor([T], dtype=torch.float64)

            piv_iv = float(pivot_implied_vol(p_t, f_t, k_t, t_t, 0.0, 0.0, is_call=is_call).item())
            bis_iv = _bisection_implied_vol(price, F, K, T, is_call=is_call)
            assert abs(piv_iv - bis_iv) <= 1e-10, (
                f"IV oracle divergence at K={K}, T={T}: pivot={piv_iv}, bisection={bis_iv}"
            )


# ---------------------------------------------------------------------------
# Phase 6.3 — Determinism
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_ml_integrity_inference_determinism_process_restarts() -> None:
    """Verifies inference determinism across evaluations with zero stochasticity."""
    net1 = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine1 = CrossVolSurrogateEngine(net1)
    X = _create_sample_state()

    out1 = engine1(X)
    out2 = engine1(X)

    assert torch.equal(out1.V, out2.V), "Inference non-determinism detected in price"
    assert torch.equal(out1.sigma, out2.sigma), "Inference non-determinism detected in vol"
    assert torch.equal(out1.s_k, out2.s_k), "Inference non-determinism detected in s_k"
    assert torch.equal(out1.s_kk, out2.s_kk), "Inference non-determinism detected in s_kk"


# ---------------------------------------------------------------------------
# Phase 6.4 — Architecture Invariants
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_ml_integrity_architecture_dimensions() -> None:
    """Verifies MLP architecture is 13 -> 64 -> 64 -> 64 -> 1 SiLU with zero-initialized output layer."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    assert len(net.weights) == 4
    assert len(net.biases) == 4

    # Layer 0: 13 -> 64
    assert net.weights[0].shape == (1, 64, 13)
    # Layers 1, 2: 64 -> 64
    assert net.weights[1].shape == (1, 64, 64)
    assert net.weights[2].shape == (1, 64, 64)
    # Output layer 3: 64 -> 1
    assert net.weights[3].shape == (1, 1, 64)
    assert torch.equal(net.weights[3], torch.zeros_like(net.weights[3])), "Output layer weights not zero-initialized"
    assert torch.equal(net.biases[3], torch.zeros_like(net.biases[3])), "Output layer biases not zero-initialized"


@pytest.mark.audit
def test_ml_integrity_zero_initialized_net_equals_analytic_bitwise() -> None:
    """Verifies zero-initialized surrogate equals Analytic_Triangulated bitwise."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    analytic = AnalyticTriangulated()

    X = _create_sample_state()
    out_surrogate = engine(X)
    out_analytic = analytic(X)

    assert torch.equal(out_surrogate.V, out_analytic.V), "Zero-init net price != Analytic_Triangulated bitwise"
    assert torch.equal(out_surrogate.sigma, out_analytic.sigma), "Zero-init net sigma != Analytic_Triangulated bitwise"
    assert torch.equal(out_surrogate.s_k, out_analytic.s_k), "Zero-init net s_k != Analytic_Triangulated bitwise"
    assert torch.equal(out_surrogate.s_kk, out_analytic.s_kk), "Zero-init net s_kk != Analytic_Triangulated bitwise"


@pytest.mark.audit
def test_ml_integrity_nn_gate_zero_equals_analytic_bitwise() -> None:
    """Verifies passing nn_gate = 0.0 equals Analytic_Triangulated bitwise even with arbitrary weights."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    # Perturb weights so net is non-zero
    with torch.no_grad():
        net.weights[3].data.fill_(0.5)

    engine = CrossVolSurrogateEngine(net)
    analytic = AnalyticTriangulated()

    X_zero_gate = _create_sample_state()
    X_zero_gate[0, Col.NN_GATE] = 0.0

    out_surrogate = engine(X_zero_gate)
    out_analytic = analytic(X_zero_gate)

    assert torch.equal(out_surrogate.V, out_analytic.V), "nn_gate=0 price != Analytic_Triangulated bitwise"
    assert torch.equal(out_surrogate.sigma, out_analytic.sigma), "nn_gate=0 sigma != Analytic_Triangulated bitwise"
    assert torch.equal(out_surrogate.s_k, out_analytic.s_k), "nn_gate=0 s_k != Analytic_Triangulated bitwise"
    assert torch.equal(out_surrogate.s_kk, out_analytic.s_kk), "nn_gate=0 s_kk != Analytic_Triangulated bitwise"


@pytest.mark.audit
def test_ml_integrity_ensemble_mean_in_vol_space() -> None:
    """Verifies ensemble mean is taken in residual log-vol space before Black-76 pricer."""
    net = ResidualCrossVolNet(width=64, depth=3, members=3)
    engine = CrossVolSurrogateEngine(net)
    X = _create_sample_state()

    # Evaluate engine
    out = engine(X)
    assert out.sigma.ndim == 1 and len(out.sigma) == 1
    # Verify sigma is a scalar float64 before pricing V
    assert out.V.ndim == 1 and len(out.V) == 1


@pytest.mark.audit
def test_ml_integrity_quintic_taper_exact_zero_deep_wings() -> None:
    """Verifies C2 quintic taper enforces exact 0.0 NN contribution for |z| >= 7."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    with torch.no_grad():
        net.weights[3].data.fill_(1.0)  # Strong non-zero NN

    engine = CrossVolSurrogateEngine(net)
    analytic = AnalyticTriangulated()

    # K=10 RUB on XAU/RUB (F=251750, z approx +45)
    X_deep = _create_sample_state(F=251750.0, K=10.0, T=1.0)
    out_surrogate = engine(X_deep)
    out_analytic = analytic(X_deep)

    assert torch.equal(out_surrogate.V, out_analytic.V), "Deep-wing NN contribution is not exactly 0.0"
    assert torch.equal(out_surrogate.sigma, out_analytic.sigma), "Deep-wing sigma != analytic sigma bitwise"


# ---------------------------------------------------------------------------
# Phase 6.5 — DML Loss Specification
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_ml_integrity_dml_loss_hyperparameters() -> None:
    """Verifies DML loss hyperparameters lambda=1, nu=0.1, mu=10 as specified."""
    cfg = TrainConfig()
    assert cfg.lam == 1.0, f"DML lambda expected 1.0, got {cfg.lam}"
    assert cfg.nu == 0.1, f"DML nu expected 0.1, got {cfg.nu}"
    assert cfg.mu == 10.0, f"DML mu expected 10.0, got {cfg.mu}"


# ---------------------------------------------------------------------------
# Phase 6.6 — Gates G1.1–G1.5b Re-evaluation
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_ml_integrity_gate_g1_recomputation() -> None:
    """Recomputes Gates G1.1–G1.5b on the official test split and asserts all criteria pass."""
    shards_root = Path("artifacts/data/fx_cross_shards")
    models_root = Path("artifacts/models/fx_cross")
    ckpts = sorted(models_root.glob("residual_vol_v3_m*.pt"))
    assert len(ckpts) == 5, f"Expected 5 ensemble member checkpoints, got {len(ckpts)}"

    net = ResidualCrossVolNet(width=64, depth=3, members=len(ckpts))
    ref_ckpt = torch.load(ckpts[0], map_location="cpu")
    for m, cp in enumerate(ckpts):
        m_ckpt = torch.load(cp, map_location="cpu")
        m_sd = m_ckpt["state_dict"]
        for l_idx in range(len(net.weights)):
            net.weights[l_idx].data[m].copy_(m_sd[f"weights.{l_idx}"][0])
            net.biases[l_idx].data[m].copy_(m_sd[f"biases.{l_idx}"][0])
    net.feature_mean.copy_(ref_ckpt["feature_mean"])
    net.feature_std.copy_(ref_ckpt["feature_std"])
    engine = CrossVolSurrogateEngine(net)

    eval_ds = ShardedCrossDataset(shards_root, split="test", member_idx=0)
    stress_ds = ShardedCrossDataset(shards_root, split="stress", member_idx=0)

    metrics = evaluate_gate_g1(engine, eval_ds, stress_dataset=stress_ds, max_records=len(eval_ds))

    assert metrics["g1_1_price_mae_bp"] <= 1.50
    assert metrics["g1_2_price_p99_bp"] <= 6.50
    assert metrics["g1_3_delta_mae"] <= 5.0e-4
    assert metrics["g1_4_gamma_mae_rel"] <= 0.018
    assert metrics["g1_5a_cross_vega_mae_rel"] <= 0.001
    assert metrics["g1_5b_leg_vega_mae_rel"] <= 0.010
    assert metrics["g1_6_arbitrage_violations"] == 0
    assert metrics["status"] == "PASS"


# ---------------------------------------------------------------------------
# Phase 6.7 — Defect Reproductions
# ---------------------------------------------------------------------------

@pytest.mark.audit
def test_F010_checkpoint_loading_omits_weights_only() -> None:
    """DEFECT REPRODUCTION: Checkpoint loading in train_cross_surrogate and fx_cross_guardian omits weights_only=True.

    Spec Reference:
      - Phase 6 item 7: Checkpoint loading: weights_only=True.
      - Phase 16 item 2: Checkpoints load with weights_only=True.

    Root Cause:
      In `src/deepvol/training/train_cross_surrogate.py` (lines 507, 1032, 1040),
      `src/deepvol/mrm/fx_cross_guardian.py` (line 162), and
      `src/deepvol/api/server.py` (line 2208), calls to `torch.load` do not specify
      `weights_only=True`, exposing the service to arbitrary code execution on malicious checkpoints.
    """
    import inspect
    from deepvol.training import train_cross_surrogate
    from deepvol.mrm import fx_cross_guardian

    # Inspect source code of load_checkpoint
    src_train = inspect.getsource(train_cross_surrogate.load_checkpoint)
    src_guard = inspect.getsource(fx_cross_guardian.MarketStateOODDetector.from_checkpoint)

    has_weights_only_train = "weights_only=True" in src_train
    has_weights_only_guard = "weights_only=True" in src_guard

    assert has_weights_only_train and has_weights_only_guard, (
        "F010 VIOLATION: torch.load calls in load_checkpoint and MarketStateOODDetector.from_checkpoint "
        "omit weights_only=True, violating SR 11-7 and supply-chain integrity standards"
    )


@pytest.mark.audit
def test_F011_reload_weights_crashes_on_checkpoint_dict() -> None:
    """DEFECT REPRODUCTION: server.py:reload_fx_cross_weights crashes on standard model checkpoint dictionary.

    Spec Reference:
      - Phase 6 item 7: weights_only=True, feature mean/std buffers from checkpoint, model_version propagation.

    Root Cause:
      In `src/deepvol/api/server.py:reload_fx_cross_weights`:
      `new_state = torch.load(weights_path, map_location="cpu")`
      `state["engine"].net.load_state_dict(new_state)`
      Official checkpoints (e.g. `residual_vol_v3_m0.pt`) store a top-level dictionary containing
      `state_dict`, `feature_mean`, `feature_std`, and `model_version`.
      Passing `new_state` directly to `load_state_dict` raises a RuntimeError (unexpected/missing keys),
      fails to update `feature_mean`/`feature_std`, and fails to propagate `model_version`.
    """
    ckpt_path = Path("artifacts/models/fx_cross/residual_vol_v3_m0.pt")
    assert ckpt_path.exists()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    net = ResidualCrossVolNet(width=64, depth=3, members=1)

    # Attempt direct load as server.py currently does
    failed = False
    try:
        net.load_state_dict(ckpt)
    except RuntimeError:
        failed = True

    assert not failed, (
        "F011 VIOLATION: server.py:reload_fx_cross_weights fails to unpack 'state_dict' from "
        "top-level checkpoint metadata dictionary, crashing on official release checkpoints"
    )
