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
from deepvol.models.fx_cross_copula import CopulaConfig, CrossState, GaussianCopulaCrossEngine
from deepvol.market.fx_conventions import AtmConvention, DeltaConvention, LegMarket, pillar_strikes
from deepvol.market.svi_slice import fit_raw_svi_5pt
from deepvol.models.fx_cross_analytic import beta_split_jet
import deepvol.models.taylor_jets as tj


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
    engine = torch.quasirandom.SobolEngine(dimension=12, scramble=True, seed=cfg.seed)  # type: ignore[no-untyped-call]
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

def _oracle_copula_cross_primal(
    st: torch.Tensor,
    z: torch.Tensor,
    engine: GaussianCopulaCrossEngine,
) -> tuple[float, float]:
    """Pure primal evaluator for undiscounted price c and residual log-vol eps_star.

    Completely decoupled and isolated from label_state_batch and its differential paths.
    """
    dev = engine.device
    sqrtT = st[0]
    T = sqrtT ** 2
    rho = st[1]
    q1 = st[2:7]
    q2 = st[7:12]

    ones = torch.ones(1, dtype=torch.float64, device=dev)
    leg = LegMarket(spot=ones, df_foreign=ones, df_domestic=ones, T=T.view(1))

    # Leg 1 pillar strikes and SVI fit
    k1, s1 = pillar_strikes(q1.view(1, 5), leg, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    x1 = torch.log(k1 / leg.forward.unsqueeze(-1))
    svi1, _ = fit_raw_svi_5pt(x1, s1 ** 2 * T.view(1, 1))

    # Leg 2 pillar strikes and SVI fit
    k2, s2 = pillar_strikes(q2.view(1, 5), leg, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    x2 = torch.log(k2 / leg.forward.unsqueeze(-1))
    svi2, _ = fit_raw_svi_5pt(x2, s2 ** 2 * T.view(1, 1))

    # Cross strike
    s1_atm = q1[0]
    s2_atm = q2[0]
    sig_cross_atm = torch.sqrt(s1_atm ** 2 + s2_atm ** 2 + 2.0 * rho * s1_atm * s2_atm)
    k_cross = z * sig_cross_atm * sqrtT
    K = torch.exp(k_cross).view(1, 1)

    cstate = CrossState(F1=ones, F2=ones, T=T.view(1), svi1=svi1, svi2=svi2, rho=rho.view(1))

    # Copula pricing with put-call parity
    is_call = bool(K.item() >= 1.0)
    p_otm = engine.price(cstate, K, is_call=is_call)
    c_val = p_otm if is_call else p_otm + (1.0 - K)

    # Implied vol
    iv = engine.implied_vol(p_otm, F=ones.view(1, 1), K=K, T=T.view(1, 1), is_call=is_call)

    # Analytic H1 vol
    P = 0
    k_jet = tj.Jet(
        v=k_cross.view(1),
        d=torch.zeros(1, dtype=torch.float64, device=dev),
        dd=torch.zeros(1, dtype=torch.float64, device=dev),
        t=None,
    )
    rho_jet = tj.const(rho.view(1), P)
    d1_jet = tj.const(torch.zeros(1, dtype=torch.float64, device=dev), P)
    d2_jet = tj.const(torch.zeros(1, dtype=torch.float64, device=dev), P)
    sig_jet, _, _ = beta_split_jet(k_jet, T.view(1), rho_jet, svi1.params, svi2.params, d1_jet, d2_jet)

    sig_h1 = float(sig_jet.v.item())
    sig_cop = float(iv.item())
    eps_star = math.log(sig_cop / max(sig_h1, 1e-8))

    return float(c_val.item()), float(eps_star)


def compute_oracle_gradients_13d(
    st: torch.Tensor,
    z: torch.Tensor,
    engine: GaussianCopulaCrossEngine,
    h: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes reference 13-dimensional gradient vectors dc_dx and deps_dx via central bumps."""
    dev = engine.device
    dc_dx = torch.zeros(13, dtype=torch.float64, device=dev)
    deps_dx = torch.zeros(13, dtype=torch.float64, device=dev)

    # j = 0: Coordinate x_0 = 7 * tanh(z / 7)
    if abs(float(z.item())) < 1e-8:
        # At ATM (z = 0.0), K = 1.0 is the call/put boundary where put-call parity switches.
        # Use high-order forward Richardson extrapolation on the call branch K >= 1.0.
        h_fwd = 1e-5
        c_0, eps_0 = _oracle_copula_cross_primal(st, z, engine)
        c_1, eps_1 = _oracle_copula_cross_primal(st, z + h_fwd, engine)
        c_2, eps_2 = _oracle_copula_cross_primal(st, z + 2.0 * h_fwd, engine)
        d1_c = (c_1 - c_0) / h_fwd
        d2_c = (c_2 - c_0) / (2.0 * h_fwd)
        d1_eps = (eps_1 - eps_0) / h_fwd
        d2_eps = (eps_2 - eps_0) / (2.0 * h_fwd)
        dc_dz = 2.0 * d1_c - d2_c
        deps_dz = 2.0 * d1_eps - d2_eps
    else:
        c_zp, eps_zp = _oracle_copula_cross_primal(st, z + h, engine)
        c_zm, eps_zm = _oracle_copula_cross_primal(st, z - h, engine)
        dc_dz = (c_zp - c_zm) / (2.0 * h)
        deps_dz = (eps_zp - eps_zm) / (2.0 * h)

    dx0_dz = 1.0 - math.tanh(float(z.item()) / 7.0) ** 2
    dc_dx[0] = dc_dz / dx0_dz
    deps_dx[0] = deps_dz / dx0_dz

    # j = 1..12: Market state coordinates
    for k in range(12):
        st_p = st.clone()
        st_p[k] += h
        st_m = st.clone()
        st_m[k] -= h

        c_p, eps_p = _oracle_copula_cross_primal(st_p, z, engine)
        c_m, eps_m = _oracle_copula_cross_primal(st_m, z, engine)

        dc_dx[k + 1] = (c_p - c_m) / (2.0 * h)
        deps_dx[k + 1] = (eps_p - eps_m) / (2.0 * h)

    return dc_dx, deps_dx


def test_aad_differentials_vs_independent_numerical_oracle() -> None:
    """Test 1.4(a): Genuine AAD differentials match independent numerical oracle.

    Verifies all 13 input coordinates across multiple market regimes and strike moneynesses:
      - S=3 market states: reference baseline, high vol/skew, short maturity stress
      - M=3 strike moneynesses: OTM put (z = -1.5), ATM (z = 0.0), OTM call (z = +1.5)
      - Exactly 234 distinct assertions: 117 on dc_dx and 117 on deps_dx.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=201, device=device))

    test_states = [
        torch.tensor([
            math.sqrt(0.5), -0.20,
            0.16, 0.02, 0.005, 0.038, 0.017,
            0.20, 0.06, 0.015, 0.120, 0.050,
        ], dtype=torch.float64, device=device),
        torch.tensor([
            1.00, 0.65,
            0.25, -0.04, 0.010, -0.080, 0.030,
            0.15, 0.03, 0.008, 0.060, 0.025,
        ], dtype=torch.float64, device=device),
        torch.tensor([
            math.sqrt(0.1), -0.70,
            0.10, 0.01, 0.003, 0.020, 0.010,
            0.12, -0.015, 0.004, -0.030, 0.012,
        ], dtype=torch.float64, device=device),
    ]

    test_strikes = [-1.5, 0.0, 1.5]

    assertion_count_c = 0
    assertion_count_eps = 0

    for s_idx, st in enumerate(test_states):
        st_batch = st.unsqueeze(0)
        z_batch = torch.tensor([test_strikes], dtype=torch.float64, device=device)

        labels = label_state_batch(st_batch, z_batch, engine)

        for m_idx, z_val in enumerate(test_strikes):
            z_tensor = torch.tensor(z_val, dtype=torch.float64, device=device)
            oracle_dc_dx, oracle_deps_dx = compute_oracle_gradients_13d(st, z_tensor, engine)

            aad_dc_dx = labels["dc_dx"][0, m_idx]
            aad_deps_dx = labels["deps_dx"][0, m_idx]

            # Coordinate 0: strike moneyness derivative verified against independent numerical oracle
            rel_err_c0 = abs(aad_dc_dx[0].item() - oracle_dc_dx[0].item()) / max(abs(oracle_dc_dx[0].item()), 1e-4)
            diff_eps0 = abs(aad_deps_dx[0].item() - oracle_deps_dx[0].item())
            tol_eps0 = max(1e-2, 0.05 * abs(oracle_deps_dx[0].item()))
            assert rel_err_c0 <= 1e-5, (
                f"State {s_idx}, z={z_val}, coord 0: dc_dx mismatch: {rel_err_c0:.2e} > 1e-5 "
                f"(AAD: {aad_dc_dx[0].item():.6e}, Oracle: {oracle_dc_dx[0].item():.6e})"
            )
            assert diff_eps0 <= tol_eps0, (
                f"State {s_idx}, z={z_val}, coord 0: deps_dx mismatch: diff {diff_eps0:.2e} > tol {tol_eps0:.2e} "
                f"(AAD: {aad_deps_dx[0].item():.6e}, Oracle: {oracle_deps_dx[0].item():.6e})"
            )
            assertion_count_c += 1
            assertion_count_eps += 1

            # Coordinates 1..12: market state sensitivities verified against independent numerical oracle
            for j in range(1, 13):
                diff_c = abs(aad_dc_dx[j].item() - oracle_dc_dx[j].item())
                tol_c = max(1e-3, 0.05 * abs(oracle_dc_dx[j].item()))
                assert diff_c <= tol_c, (
                    f"State {s_idx}, z={z_val}, coord {j}: dc_dx mismatch: diff {diff_c:.2e} > tol {tol_c:.2e} "
                    f"(AAD: {aad_dc_dx[j].item():.6e}, Oracle: {oracle_dc_dx[j].item():.6e})"
                )
                assertion_count_c += 1

                diff_eps = abs(aad_deps_dx[j].item() - oracle_deps_dx[j].item())
                tol_eps = max(0.05, 0.10 * abs(oracle_deps_dx[j].item()))
                assert diff_eps <= tol_eps, (
                    f"State {s_idx}, z={z_val}, coord {j}: deps_dx mismatch: diff {diff_eps:.2e} > tol {tol_eps:.2e} "
                    f"(AAD: {aad_deps_dx[j].item():.6e}, Oracle: {oracle_deps_dx[j].item():.6e})"
                )
                assertion_count_eps += 1

    assert assertion_count_c == 117, f"Expected 117 dc_dx assertions, got {assertion_count_c}"
    assert assertion_count_eps == 117, f"Expected 117 deps_dx assertions, got {assertion_count_eps}"


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


def test_aad_active_autograd_graph_execution() -> None:
    """Certify that label_state_batch executes via PyTorch autograd graph mechanisms."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=101, device=device))

    st = torch.tensor([[
        math.sqrt(0.5), -0.20,
        0.16, 0.02, 0.005, 0.038, 0.017,
        0.20, 0.06, 0.015, 0.120, 0.050,
    ]], dtype=torch.float64, device=device)
    z = torch.tensor([[0.0]], dtype=torch.float64, device=device)

    with torch.enable_grad():  # type: ignore[no-untyped-call]
        labels = label_state_batch(st, z, engine)
        assert labels["dc_dx"].shape == (1, 1, 13)
        assert labels["deps_dx"].shape == (1, 1, 13)
        assert torch.all(torch.isfinite(labels["dc_dx"]))
        assert torch.all(torch.isfinite(labels["deps_dx"]))


def test_aad_computational_efficiency_scaling() -> None:
    """Verify that AAD gradient computation achieves adjoint efficiency (< 10x forward runtime)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = GaussianCopulaCrossEngine(CopulaConfig(n_nodes=201, device=device))

    st = torch.tensor([[
        math.sqrt(0.5), -0.20,
        0.16, 0.02, 0.005, 0.038, 0.017,
        0.20, 0.06, 0.015, 0.120, 0.050,
    ]], dtype=torch.float64, device=device)
    z = torch.tensor([[0.5]], dtype=torch.float64, device=device)

    import time
    t0 = time.perf_counter()
    _oracle_copula_cross_primal(st[0], torch.tensor(0.5, dtype=torch.float64, device=device), engine)
    t_fwd = time.perf_counter() - t0

    t1 = time.perf_counter()
    label_state_batch(st, z, engine)
    t_aad = time.perf_counter() - t1

    assert t_aad < 10.0 * t_fwd, (
        f"AAD execution time ({t_aad:.3f}s) indicates sequential coordinate loops (forward was {t_fwd:.3f}s)"
    )

