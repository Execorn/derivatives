"""tests/test_train_cross_surrogate.py — Unit and integration tests for Task 2.2 & 2.3.

Verifies:
- Test (f): ShardedCrossDataset memory-mapped streaming and state-level bootstrap resampling per member.
- Test (g): Composite Sobolev DML loss calculation with double backward (create_graph=True).
- Test (h): Taylor-mode Durrleman hinge penalty evaluation on collocation strikes.
- Test (i): Bitwise checkpoint reload test, model_version reproducibility, and model_card.json schema.
- Test (j): Fast smoke training CLI run (--steps 5 --members 2).
- Test (k): Ledoit-Wolf 12-D OOD statistics fitting and Mahalanobis distance evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine, ResidualCrossVolNet
from deepvol.training.train_cross_surrogate import (
    SHARD_DTYPE,
    ShardedCrossDataset,
    TrainConfig,
    compute_model_version,
    dml_loss,
    durrleman_hinge,
    fit_ood_statistics,
    load_checkpoint,
    main,
    save_checkpoint,
)


@pytest.fixture
def device() -> str:
    """Fixture providing available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _create_mock_shards(root: Path, n_states: int = 4, strikes_per_state: int = 64) -> None:
    """Creates synthetic binary .npy shards and manifest.json for testing."""
    for split in ("train", "val", "test", "stress"):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        n_records = n_states * strikes_per_state
        arr = np.zeros(n_records, dtype=SHARD_DTYPE)

        # Populate synthetic features
        for s in range(n_states):
            st_slice = slice(s * strikes_per_state, (s + 1) * strikes_per_state)
            sqrtT = 1.0 + 0.1 * s
            rho = -0.2 + 0.05 * s
            z_vals = np.linspace(-3.0, 3.0, strikes_per_state)
            z_in = 7.0 * np.tanh(z_vals / 7.0)

            for i_m, idx in enumerate(range(s * strikes_per_state, (s + 1) * strikes_per_state)):
                x_vec = np.zeros(13, dtype=np.float32)
                x_vec[0] = z_in[i_m]
                x_vec[1] = sqrtT
                x_vec[2] = rho
                x_vec[3:8] = [0.20, -0.05, 0.02, -0.08, 0.04]  # xi1
                x_vec[8:13] = [0.25, -0.03, 0.03, -0.05, 0.05]  # xi2

                arr["x"][idx] = x_vec
                arr["c"][idx] = 0.05 + 0.001 * i_m
                arr["dc_dx"][idx] = np.full(13, 0.01)
                arr["eps_star"][idx] = 0.02 * np.sin(z_vals[i_m])
                arr["deps_dx"][idx] = np.full(13, 0.005)
                arr["sigma_h1"][idx] = 0.22
                arr["sigma_cop"][idx] = 0.22 * np.exp(arr["eps_star"][idx])

        shard_path = split_dir / "shard_0000.npy"
        np.save(shard_path, arr)

    manifest = {
        "sobol_seed": 20261011,
        "train_mean": [0.0] * 13,
        "train_std": [1.0] * 13,
        "git_hash": "24977e7",
        "shards": {"train": ["train/shard_0000.npy"]},
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def test_sharded_dataset_mmap_and_bootstrap_resampling(tmp_path: Path) -> None:
    """Test (f): ShardedCrossDataset memory-mapped streaming and state-level bootstrap resampling."""
    shards_dir = tmp_path / "shards"
    _create_mock_shards(shards_dir, n_states=8, strikes_per_state=64)

    # Initialize datasets for member 0 and member 1
    ds0 = ShardedCrossDataset(shards_dir, split="train", member_idx=0, seed=42)
    ds1 = ShardedCrossDataset(shards_dir, split="train", member_idx=1, seed=42)

    total_expected = 8 * 64
    assert len(ds0) == total_expected
    assert len(ds1) == total_expected

    # Test random access item
    rec0 = ds0[0]
    assert rec0["x"].shape == (13,)
    assert rec0["x"].dtype == torch.float32
    assert rec0["c"].dtype == torch.float64
    assert rec0["sigma_h1"].dtype == torch.float64

    # Verify state-level bootstrap universes exist and differ between seeds/members
    states_m0 = ds0.get_bootstrap_state_indices()
    states_m1 = ds1.get_bootstrap_state_indices()
    assert len(states_m0) == 8
    assert len(states_m1) == 8
    assert not np.array_equal(states_m0, states_m1), "Different members must have distinct bootstrap state samples"

    # Verify batch generator returns tensor batches of specified batch size
    batch = ds0.sample_batch(batch_size=32, device="cpu")
    assert batch["x"].shape == (32, 13)
    assert batch["c"].shape == (32,)
    assert batch["dc_dx"].shape == (32, 13)
    assert batch["eps_star"].shape == (32,)
    assert batch["sigma_h1"].shape == (32,)


def test_composite_sobolev_dml_loss(device: str) -> None:
    """Test (g): Composite Sobolev DML loss calculation with double backward (create_graph=True)."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    B = 32

    # Synthetic batch
    x = torch.randn(B, 13, dtype=torch.float32, device=device)
    x[:, 0] = 7.0 * torch.tanh(x[:, 0] / 7.0)  # valid squashed moneyness
    x[:, 1] = torch.clamp(x[:, 1].abs(), min=0.1, max=2.0)  # sqrtT
    x[:, 2] = torch.clamp(x[:, 2], min=-0.8, max=0.8)  # rho
    x[:, 3] = 0.20  # s1
    x[:, 8] = 0.25  # s2

    batch = {
        "x": x,
        "c": torch.full((B,), 0.05, dtype=torch.float64, device=device),
        "dc_dx": torch.full((B, 13), 0.01, dtype=torch.float64, device=device),
        "eps_star": torch.full((B,), 0.01, dtype=torch.float64, device=device),
        "deps_dx": torch.full((B, 13), 0.005, dtype=torch.float64, device=device),
        "sigma_h1": torch.full((B,), 0.22, dtype=torch.float64, device=device),
    }

    stats = {
        "var_c": torch.tensor(0.01, dtype=torch.float64, device=device),
        "mean_dc_dx_sq": torch.full((13,), 0.001, dtype=torch.float64, device=device),
        "var_eps": torch.tensor(0.001, dtype=torch.float64, device=device),
    }

    cfg = TrainConfig(lam=1.0, nu=0.1, mu=0.0)
    loss, metrics = dml_loss(net, batch, stats, cfg)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert "loss_price" in metrics
    assert "loss_diff" in metrics
    assert "loss_vol" in metrics

    # Test backward pass populates gradients across all network parameters
    loss.backward()
    for w in net.weights:
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()
        assert not torch.equal(w.grad, torch.zeros_like(w.grad))


def test_durrleman_hinge_collocation(device: str) -> None:
    """Test (h): Taylor-mode Durrleman hinge penalty evaluation on collocation strikes."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1).to(device)
    engine = CrossVolSurrogateEngine(net).to(device)

    S_cnt = 2
    M_cnt = 32

    # Arbitrage-free market states
    states = torch.zeros(S_cnt, 12, dtype=torch.float64, device=device)
    states[:, 0] = 1.0  # sqrtT
    states[:, 1] = -0.20  # rho
    # SVI slices for leg 1 and leg 2: (a, b, rho_s, m, s)
    states[:, 2:7] = torch.tensor([0.025, 0.08, -0.15, 0.02, 0.05], dtype=torch.float64, device=device)
    states[:, 7:12] = torch.tensor([0.040, 0.12, 0.35, -0.01, 0.08], dtype=torch.float64, device=device)

    k_colloc = torch.linspace(-0.5, 0.5, M_cnt, dtype=torch.float64, device=device).unsqueeze(0).expand(S_cnt, -1)

    hinge = durrleman_hinge(engine, states, k_colloc)
    assert hinge.dim() == 0
    assert torch.isfinite(hinge)
    # Zero-initialized net on arbitrage-free SVI legs has 0 butterfly violations
    assert hinge.item() <= 1e-12


def test_checkpoint_reload_and_model_card(tmp_path: Path) -> None:
    """Test (i): Bitwise checkpoint reload test, model_version reproducibility, and model_card.json schema."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0.0, 0.05)

    ckpt_path = tmp_path / "residual_vol_v3_m0.pt"
    git_hash = "24977e7d6928e08cf4410a51c411516e5461c3bc"
    manifest_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    save_checkpoint(
        path=ckpt_path,
        net=net,
        member_idx=0,
        step=24000,
        train_loss=0.0125,
        val_loss=0.0142,
        git_hash=git_hash,
        data_manifest_hash=manifest_hash,
    )
    assert ckpt_path.exists()

    # Bitwise reload
    net_reloaded = ResidualCrossVolNet(width=64, depth=3, members=1)
    meta = load_checkpoint(ckpt_path, net_reloaded)

    for p_orig, p_reload in zip(net.parameters(), net_reloaded.parameters()):
        assert torch.equal(p_orig, p_reload), "Reloaded parameter tensor was not bitwise identical"

    # Model version reproducibility: sha256(weights)[:12] + git short hash
    version = compute_model_version(net.state_dict(), git_hash)
    assert meta["model_version"] == version
    assert "+" in version
    digest, git_short = version.split("+")
    assert len(digest) == 12
    assert len(git_short) == 7


def test_fast_smoke_training_cli(tmp_path: Path) -> None:
    """Test (j): Fast smoke training CLI run (--steps 5 --members 2)."""
    shards_dir = tmp_path / "shards"
    _create_mock_shards(shards_dir, n_states=4, strikes_per_state=64)
    out_dir = tmp_path / "models"

    ret = main([
        "--shards", str(shards_dir),
        "--steps", "5",
        "--members", "2",
        "--batch-size", "16",
        "--out", str(out_dir),
    ])
    assert ret == 0

    assert (out_dir / "residual_vol_v3_m0.pt").exists()
    assert (out_dir / "residual_vol_v3_m1.pt").exists()
    assert (out_dir / "model_card.json").exists()

    with open(out_dir / "model_card.json", "r", encoding="utf-8") as f:
        card = json.load(f)

    # Validate model card schema keys
    for req_key in (
        "model_name",
        "model_version",
        "git_hash",
        "data_manifest_hash",
        "architecture",
        "ensemble",
        "feature_statistics",
        "ood_statistics",
        "gate_g1_metrics",
        "file_checksums",
        "created_at",
    ):
        assert req_key in card, f"Missing required model card key: {req_key}"


def test_fit_ood_statistics() -> None:
    """Test (k): Ledoit-Wolf 12-D OOD statistics fitting and Mahalanobis scoring."""
    torch.manual_seed(123)
    train_states = torch.randn(100, 12, dtype=torch.float64)

    stats = fit_ood_statistics(train_states)

    assert "mean_12d" in stats
    assert "cov_inv_12d" in stats
    assert "q995_mahalanobis" in stats
    assert "box_lo" in stats
    assert "box_hi" in stats

    assert stats["mean_12d"].shape == (12,)
    assert stats["cov_inv_12d"].shape == (12, 12)
    assert stats["q995_mahalanobis"].item() > 0.0

    # In-distribution test points vs out-of-distribution point
    test_id = torch.randn(1, 12, dtype=torch.float64)
    diff_id = test_id - stats["mean_12d"]
    d_m_id = torch.sqrt(diff_id @ stats["cov_inv_12d"] @ diff_id.t()).item()

    test_ood = test_id + 50.0  # huge shock
    diff_ood = test_ood - stats["mean_12d"]
    d_m_ood = torch.sqrt(diff_ood @ stats["cov_inv_12d"] @ diff_ood.t()).item()

    assert d_m_ood > d_m_id
    assert d_m_ood > stats["q995_mahalanobis"].item()
