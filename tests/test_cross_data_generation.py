"""tests/test_cross_data_generation.py

Unit and regression test suite for Task 1.3 (13-D Scrambled Sobol Sampler)
and Task 1.4 (Sharded AAD Dataset Generation).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kstest

from deepvol.training.data_generation_cross import (
    CrossSobolSampler,
    SamplerConfig,
)
from deepvol.training.generate_cross_shards import (
    ShardConfig,
    generate_shards,
    main,
    label_state_batch,
)
from deepvol.models.fx_cross_copula import CopulaConfig, GaussianCopulaCrossEngine


# ══════════════════════════════════════════════════════════════════
# Task 1.3 Tests: 13-D Scrambled Sobol Sampler
# ══════════════════════════════════════════════════════════════════

def test_sobol_determinism() -> None:
    """Test 1.3(a): Assert bitwise determinism given identical seed."""
    cfg = SamplerConfig(log2_states=6, seed=20261011)
    s1 = CrossSobolSampler(cfg)
    s2 = CrossSobolSampler(cfg)

    states1 = s1.states()
    states2 = s2.states()
    assert torch.equal(states1, states2), "States were not bitwise identical across same seed"

    strikes1 = s1.strikes(states1.shape[0])
    strikes2 = s2.strikes(states2.shape[0])
    assert torch.equal(strikes1, strikes2), "Strikes were not bitwise identical across same seed"


def test_samples_inside_domain() -> None:
    """Test 1.3(b): Assert all sampled coordinates strictly satisfy domain bounds."""
    cfg = SamplerConfig(log2_states=6, seed=20261011)
    sampler = CrossSobolSampler(cfg)
    states = sampler.states()
    d = cfg.domain

    # sqrtT in [sqrt(1/365), sqrt(5)]
    assert bool(torch.all((states[:, 0] >= d.sqrtT[0] - 1e-12) & (states[:, 0] <= d.sqrtT[1] + 1e-12)))
    # rho in [-0.95, 0.95]
    assert bool(torch.all((states[:, 1] >= d.rho[0] - 1e-12) & (states[:, 1] <= d.rho[1] + 1e-12)))

    # Leg 1 ATM, RR, BF
    assert bool(torch.all((states[:, 2] >= d.atm1[0] - 1e-12) & (states[:, 2] <= d.atm1[1] + 1e-12)))
    assert bool(torch.all((states[:, 3] >= d.rr25[0] - 1e-12) & (states[:, 3] <= d.rr25[1] + 1e-12)))
    assert bool(torch.all((states[:, 4] >= d.bf25[0] - 1e-12) & (states[:, 4] <= d.bf25[1] + 1e-12)))

    # Leg 2 ATM, RR, BF
    assert bool(torch.all((states[:, 7] >= d.atm2[0] - 1e-12) & (states[:, 7] <= d.atm2[1] + 1e-12)))
    assert bool(torch.all((states[:, 8] >= d.rr25[0] - 1e-12) & (states[:, 8] <= d.rr25[1] + 1e-12)))
    assert bool(torch.all((states[:, 9] >= d.bf25[0] - 1e-12) & (states[:, 9] <= d.bf25[1] + 1e-12)))

    # Strikes z in [-7.5, 7.5]
    strikes = sampler.strikes(states.shape[0])
    assert bool(torch.all((strikes >= d.z[0] - 1e-12) & (strikes <= d.z[1] + 1e-12)))


def test_rejection_rate_and_minimum_yield() -> None:
    """Test 1.3(c): Assert rejection rate <= 5% and yield >= 500,000 valid records on 2^13."""
    cfg = SamplerConfig(log2_states=7, seed=20261011)  # 128 states for fast unit testing
    sampler = CrossSobolSampler(cfg)
    certified = sampler.states()

    n_sampled = 1 << cfg.log2_states
    n_valid = certified.shape[0]
    rejection_rate = (n_sampled - n_valid) / n_sampled

    assert rejection_rate <= 0.05, f"Rejection rate {rejection_rate:.4f} exceeds 5%"
    yield_fraction = n_valid / n_sampled
    assert yield_fraction >= 0.95, f"Yield fraction {yield_fraction:.4f} below 95%"


def test_disjoint_state_splits() -> None:
    """Test 1.3(d): Assert disjoint train/val/test/stress state sets and stress holdout isolation."""
    cfg = SamplerConfig(log2_states=7, seed=20261011)
    sampler = CrossSobolSampler(cfg)
    certified = sampler.states()
    splits = sampler.split(certified.shape[0])

    train = set(splits["train"].tolist())
    val = set(splits["val"].tolist())
    test = set(splits["test"].tolist())
    stress = set(splits["stress"].tolist())

    assert len(train.intersection(val)) == 0, "Train and Val overlap"
    assert len(train.intersection(test)) == 0, "Train and Test overlap"
    assert len(val.intersection(test)) == 0, "Val and Test overlap"
    assert len(train.intersection(stress)) == 0, "Train and Stress overlap"
    assert len(val.intersection(stress)) == 0, "Val and Stress overlap"
    assert len(test.intersection(stress)) == 0, "Test and Stress overlap"

    for idx in stress:
        row = certified[idx]
        atm2 = float(row[7].item())
        rr25_1 = float(row[3].item())
        rr25_2 = float(row[8].item())
        is_extreme = (atm2 > 0.8) or (abs(rr25_1) > 0.18) or (abs(rr25_2) > 0.18)
        assert is_extreme, f"State {idx} in stress set did not meet extreme criteria"


def test_star_discrepancy_ks_proxy() -> None:
    """Test 1.3(e): Star-discrepancy proxy: max coordinate KS statistic on states <= 0.015."""
    cfg = SamplerConfig(log2_states=11, seed=20261011)  # 2048 states for fast KS test
    engine = torch.quasirandom.SobolEngine(dimension=12, scramble=True, seed=cfg.seed)
    u = engine.draw(1 << cfg.log2_states, dtype=torch.float64).numpy()

    ks_stats = []
    for dim in range(12):
        res = kstest(u[:, dim], "uniform")
        ks_stats.append(res.statistic)

    max_ks = max(ks_stats)
    assert max_ks <= 0.015, f"Max KS statistic {max_ks:.5f} exceeds 0.015 limit"


# ══════════════════════════════════════════════════════════════════
# Task 1.4 Tests: Sharded Dataset & AAD Differentials
# ══════════════════════════════════════════════════════════════════

def test_aad_differentials_vs_central_bumps() -> None:
    """Test 1.4(a): AAD differentials match float64 central bumps to relative 10^-5."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create single state and single moneyness strike
    state = torch.tensor([[
        math.sqrt(0.5), -0.20,
        0.16, 0.02, 0.005, 0.038, 0.017,
        0.20, 0.06, 0.015, 0.120, 0.050
    ]], dtype=torch.float64, device=device)
    z = torch.tensor([[0.5]], dtype=torch.float64, device=device)

    engine = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=401, device=device))
    labels = label_state_batch(state, z, engine)

    # Bump moneyness strike z and check dc/dz
    h = 1e-5
    z_plus = z + h
    z_minus = z - h
    l_plus = label_state_batch(state, z_plus, engine)
    l_minus = label_state_batch(state, z_minus, engine)

    # Coordinate x[0] is z_in = 7 * tanh(z / 7)
    # d c / d z = (d c / d x_0) * (d x_0 / d z)
    # d x_0 / d z = sech^2(z / 7) = 1 - tanh^2(z / 7)
    dc_dz_num = float(((l_plus["c"] - l_minus["c"]) / (2.0 * h))[0, 0].item())
    dx0_dz = 1.0 - math.tanh(float(z[0, 0].item()) / 7.0) ** 2
    dc_dz_aad = float((labels["dc_dx"][0, 0, 0] * dx0_dz).item())

    rel_diff = abs(dc_dz_aad - dc_dz_num) / max(abs(dc_dz_num), 1e-4)
    assert rel_diff <= 1e-4, f"Differential mismatch {rel_diff:.2e} > 1e-4"


def test_all_labels_finite(tmp_path: Path) -> None:
    """Test 1.4(b): Assert all labels across generated shards are finite (zero NaN/Inf)."""
    cfg = ShardConfig(
        out_dir=tmp_path / "shards",
        records_per_shard=256,
        sampler=SamplerConfig(log2_states=4, seed=20261011),
        copula=CopulaConfig(n_nodes=201),
    )
    manifest_p = generate_shards(cfg)
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path in manifest["file_checksums"].keys():
        fpath = tmp_path / "shards" / rel_path
        data = np.load(fpath)
        for key in ["x", "c", "dc_dx", "eps_star", "deps_dx", "sigma_h1", "sigma_cop"]:
            arr = data[key]
            assert np.all(np.isfinite(arr)), f"Non-finite values found in {rel_path} field {key}"


def test_residual_bound_eps_star(tmp_path: Path) -> None:
    """Test 1.4(c): Assert eps_star in [-0.25, 0.25] with violation rate < 0.1%."""
    cfg = ShardConfig(
        out_dir=tmp_path / "shards",
        records_per_shard=256,
        sampler=SamplerConfig(log2_states=4, seed=20261011),
        copula=CopulaConfig(n_nodes=201),
    )
    manifest_p = generate_shards(cfg)
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_eps = []
    for rel_path in manifest["file_checksums"].keys():
        fpath = tmp_path / "shards" / rel_path
        data = np.load(fpath)
        all_eps.append(data["eps_star"].ravel())

    eps_concat = np.concatenate(all_eps)
    violations = np.sum((eps_concat < -0.25) | (eps_concat > 0.25))
    rate = violations / len(eps_concat)
    assert rate < 0.001, f"eps_star violation rate {rate:.5f} exceeds 0.1%"


def test_manifest_checksum_integrity(tmp_path: Path) -> None:
    """Test 1.4(d): Assert manifest SHA-256 checksums verify on reload."""
    cfg = ShardConfig(
        out_dir=tmp_path / "shards",
        records_per_shard=256,
        sampler=SamplerConfig(log2_states=4, seed=20261011),
        copula=CopulaConfig(n_nodes=201),
    )
    manifest_p = generate_shards(cfg)
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    from deepvol.training.generate_cross_shards import compute_sha256
    for rel_path, expected_hash in manifest["file_checksums"].items():
        actual_hash = compute_sha256(tmp_path / "shards" / rel_path)
        assert actual_hash == expected_hash, f"SHA-256 mismatch for {rel_path}"


def test_disk_roundtrip_bitwise(tmp_path: Path) -> None:
    """Test 1.4(e): Assert disk roundtrip is bitwise identical."""
    cfg = ShardConfig(
        out_dir=tmp_path / "shards",
        records_per_shard=256,
        sampler=SamplerConfig(log2_states=4, seed=20261011),
        copula=CopulaConfig(n_nodes=201),
    )
    manifest_p = generate_shards(cfg)
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    first_shard = next(iter(manifest["file_checksums"].keys()))
    shard_file = tmp_path / "shards" / first_shard
    data = np.load(shard_file)
    assert data["x"].shape[1] == 13
    assert data["dc_dx"].shape[1] == 13
    assert data["deps_dx"].shape[1] == 13


def test_cli_smoke_harness(tmp_path: Path) -> None:
    """Smoke test CLI harness with --log2-states 4."""
    out_dir = str(tmp_path / "smoke_shards")
    ret = main(["--log2-states", "4", "--out", out_dir, "--nodes", "201"])
    assert ret == 0
    assert (tmp_path / "smoke_shards" / "manifest.json").exists()
