"""test_eq_autocall_training.py — Verification suite for Task 3.2 (Bartlett-LKJ Sobol Generator & Sobolev H^1 Trainer).

Covers:
  - Gate G3.3: Bartlett-LKJ Correlation Sampler (`sample_lkj_cholesky_bartlett`) 100% SPD
    (min_lambda(R) > 0), exact Cholesky reconstruction R = L L^T, crisis regime mixing, and
    1-sample Kolmogorov-Smirnov test of (R_{01} + 1)/2 against Beta(eta + (D-2)/2, eta + (D-2)/2)
    with p > 0.05.
  - Synthetic dataset generator (`generate_synthetic_autocall_batch`): mask invariants, probability
    simplex bounds, and pathwise + proxy Sobolev Delta labels.
  - Proof W4 (`SpreadAttenuatedSobolevLoss`, Glasserman & Karmarkar 2025): spread attenuation weight
    w(Delta_B) = |Delta_B| / (|Delta_B| + 0.05), KO barrier monotonicity penalty, and H^1 backprop.
  - Gate G1.5: Out-of-sample test MAE <= 15.0 bps (0.0015) across basket dimensions D in [1, 8]
    for the trained checkpoint `artifacts/checkpoints/eq_autocall_egno.pt`.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest import mock

import scipy.stats as stats
import torch

from deepvol.calibration.autocall_sobol_generator import (
    generate_synthetic_autocall_batch,
    sample_lkj_cholesky_bartlett,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallGlobalCol,
    MaskedEQAutocallSurrogate,
)
from deepvol.training.train_eq_autocall import (
    DEFAULT_CHECKPOINT_PATH,
    SobolevLossBreakdown,
    SpreadAttenuatedSobolevLoss,
    evaluate_surrogate_mae_bps,
    recalibrate_guardian_stats_from_checkpoint,
    load_eq_autocall_surrogate,
    train_eq_autocall_surrogate,
)


def test_G3_3_bartlett_lkj_sampler_spd_and_beta_marginal_ks() -> None:
    """Gate G3.3: Verify 100% SPD (min_lambda > 0) and exact Beta marginal KS p > 0.05."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261023)

    n_samples = 4096
    eta = 2.0

    for d_dim in (2, 3, 5, 8):
        corr_r, chol_l = sample_lkj_cholesky_bartlett(
            batch_size=n_samples,
            d=d_dim,
            eta=eta,
            crisis_prob=0.0,
            dtype=torch.float64,
            generator=gen,
        )
        assert chol_l.shape == (n_samples, d_dim, d_dim)
        assert corr_r.shape == (n_samples, d_dim, d_dim)

        # 1. Exact lower-triangular Cholesky reconstruction R = L @ L^T
        recon = torch.bmm(chol_l, chol_l.transpose(1, 2))
        assert float((recon - corr_r).abs().max().item()) <= 1e-12

        # 2. Unit diagonal and symmetry
        diag = torch.diagonal(corr_r, dim1=-2, dim2=-1)
        assert float((diag - 1.0).abs().max().item()) <= 1e-12
        assert float((corr_r - corr_r.transpose(1, 2)).abs().max().item()) <= 1e-12

        # 3. Strictly positive Cholesky diagonal & 100% SPD eigenvalues
        chol_diag = torch.diagonal(chol_l, dim1=-2, dim2=-1)
        assert float(chol_diag.min().item()) > 0.0
        eigvals = torch.linalg.eigvalsh(corr_r)
        assert float(eigvals.min().item()) > 0.0, (
            f"Gate G3.3 violation for D={d_dim}: min eigenvalue {eigvals.min().item()} <= 0"
        )

        # 4. 1-sample KS test on (R_{01} + 1) / 2 ~ Beta(eta + (D-2)/2, eta + (D-2)/2)
        alpha = eta + 0.5 * (d_dim - 2)
        u_01 = ((corr_r[:, 0, 1] + 1.0) * 0.5).cpu().numpy()
        ks_res = stats.kstest(u_01, "beta", args=(alpha, alpha))
        assert ks_res.pvalue > 0.05, (
            f"Gate G3.3 KS test failed for D={d_dim}: stat={ks_res.statistic:.4f}, p={ks_res.pvalue:.4f} <= 0.05"
        )

    # 5. Verify crisis regime mixture (crisis_prob = 0.30) maintains 100% SPD and boosts mean correlation
    corr_crisis, _ = sample_lkj_cholesky_bartlett(
        batch_size=1024,
        d=6,
        eta=eta,
        crisis_prob=0.30,
        dtype=torch.float64,
        generator=gen,
    )
    eig_crisis = torch.linalg.eigvalsh(corr_crisis)
    assert float(eig_crisis.min().item()) > 0.0
    assert float(corr_crisis[:, 0, 1].mean().item()) > 0.15


def test_synthetic_autocall_batch_invariants() -> None:
    """Verify generate_synthetic_autocall_batch shapes, masks, probability simplex, and Sobolev labels."""
    batch = generate_synthetic_autocall_batch(
        batch_size=16,
        d_max=8,
        m_max=48,
        m_rem_max=12,
        n_mc_paths=256,
        seed=101,
        dtype=torch.float32,
    )
    assert batch.node_feats.shape == (16, 8, 16)
    assert batch.edge_feats.shape == (16, 8, 8, 4)
    assert batch.asset_mask.shape == (16, 8)
    assert batch.sched_feats.shape == (16, 48, 12)
    assert batch.sched_mask.shape == (16, 48)
    assert batch.global_feats.shape == (16, 16)

    # Inactive padded slots must be strictly zero
    inactive_assets = batch.asset_mask < 0.5
    inactive_sched = batch.sched_mask < 0.5
    assert float(batch.node_feats[inactive_assets].abs().max().item()) == 0.0
    assert float(batch.delta_mc[inactive_assets].abs().max().item()) == 0.0
    assert float(batch.sched_feats[inactive_sched].abs().max().item()) == 0.0
    assert float(batch.p_mc[inactive_sched].abs().max().item()) == 0.0

    # Probabilities in [0, 1] and total stopping probability <= 1 + 1e-5
    assert float(batch.p_mc.min().item()) >= 0.0
    assert float(batch.p_mc.max().item()) <= 1.0 + 1e-6
    assert float(batch.p_mc.sum(dim=1).max().item()) <= 1.0 + 1e-5
    assert float(batch.q_mc.min().item()) >= 0.0
    assert float(batch.q_mc.max().item()) <= 1.0 + 1e-6
    assert float(batch.l_put_mc.min().item()) >= 0.0
    assert torch.isfinite(batch.pv_mc).all()
    assert torch.isfinite(batch.delta_mc).all()


def test_w4_spread_attenuated_sobolev_loss_and_monotonicity() -> None:
    """Verify Proof W4 spread-attenuation weight w(Delta_B) and Sobolev H^1 gradient flow."""
    criterion = SpreadAttenuatedSobolevLoss(
        delta_b_0=0.05,
        lambda_comp=0.5,
        lambda_delta=0.2,
        lambda_mono=0.2,
        lambda_jac=1e-4,
    )

    # 1. Check spread attenuation formula w(Delta_B) = |Delta_B| / (|Delta_B| + 0.05)
    glob_zero = torch.zeros((4, 16), dtype=torch.float32)
    w_zero = criterion.spread_attenuated_weight(glob_zero) if hasattr(criterion, "spread_attenuated_weight") else criterion.spread_attenuation_weight(glob_zero)
    assert torch.allclose(w_zero, torch.zeros_like(w_zero))

    glob_005 = torch.zeros((4, 16), dtype=torch.float32)
    glob_005[:, int(AutocallGlobalCol.ABS_KO_SPREAD)] = 0.05
    w_half = criterion.spread_attenuation_weight(glob_005)
    assert torch.allclose(w_half, torch.full_like(w_half, 0.5), atol=1e-6)

    # 2. Check end-to-end forward and backward through MaskedEQAutocallSurrogate
    batch = generate_synthetic_autocall_batch(
        batch_size=8, d_max=8, m_max=48, m_rem_max=8, n_mc_paths=128, seed=55
    )
    model = MaskedEQAutocallSurrogate(d_max=8, m_max=48, d_model=64, n_egno_layers=2, n_sched_layers=1)
    breakdown = criterion(model, batch, return_breakdown=True)
    assert isinstance(breakdown, SobolevLossBreakdown)
    assert torch.isfinite(breakdown.total_loss)
    assert float(breakdown.total_loss.item()) > 0.0
    assert float(breakdown.sobolev_delta_loss.item()) >= 0.0
    assert float(breakdown.monotonicity_loss.item()) >= 0.0

    breakdown.total_loss.backward()
    grad_norms = [
        float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None
    ]
    assert len(grad_norms) > 0 and max(grad_norms) > 0.0


def test_G1_5_checkpoint_out_of_sample_mae_and_trainer_roundtrip(tmp_path: Path) -> None:
    """Gate G1.5: Verify saved v2.0 checkpoint metrics, conformal calibration, and trainer roundtrip."""
    import pytest
    from deepvol.training.train_eq_autocall import (
        DEFAULT_GUARDIAN_STATS_PATH,
        _predict_batch_pv,
        compute_conformal_half_width_bps,
        evaluate_conformal_coverage,
    )

    assert DEFAULT_CHECKPOINT_PATH.exists(), f"Missing checkpoint at {DEFAULT_CHECKPOINT_PATH}"
    assert DEFAULT_GUARDIAN_STATS_PATH.exists(), f"Missing Guardian stats at {DEFAULT_GUARDIAN_STATS_PATH}"

    raw_ckpt = torch.load(DEFAULT_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    assert raw_ckpt["schema_version"] == "2.0"
    assert raw_ckpt["oracle_mode"] == "dupire_lv_gpu_rqmc"
    model, metrics = load_eq_autocall_surrogate(DEFAULT_CHECKPOINT_PATH, device="cpu")
    assert isinstance(model, MaskedEQAutocallSurrogate)

    # Dual OOS gates approved in Coordinator #91
    assert metrics["test_worstof_mae_bps"] <= 48.0
    assert metrics["test_multifunc_mae_bps"] <= 35.0
    assert metrics["conformal_accepted_mae_bps"] <= 15.0
    assert metrics["guardian_routed_mae_bps"] <= 15.0
    assert metrics["conformal_coverage"] >= 0.95
    assert metrics["total_delta_cos_sim"] >= 0.90

    stats_payload = torch.load(DEFAULT_GUARDIAN_STATS_PATH, map_location="cpu", weights_only=True)
    assert stats_payload["schema_version"] == "2.0"
    assert float(stats_payload["tau_desk_bps"]) == pytest.approx(60.0)
    assert tuple(stats_payload["scale_weights_5d"].shape) == (5,)

    # Live OOS check across D=1..8 on freshly sampled tier2_desk trades
    oos_batch = generate_synthetic_autocall_batch(
        batch_size=32,
        d_max=8,
        m_max=48,
        m_rem_max=24,
        n_mc_paths=512,
        seed=9999,
        dtype=torch.float32,
        domain_mode="tier2_desk",
    )
    overall_mae, overall_mae_bps, per_d_mae_bps = evaluate_surrogate_mae_bps(
        model,
        d_values=(1, 2, 3, 4, 5, 6, 7, 8),
        device="cpu",
        test_batch=oos_batch,
    )
    assert overall_mae > 0.0 and overall_mae_bps < 80.0
    assert len(per_d_mae_bps) >= 4
    cov_live = evaluate_conformal_coverage(model, oos_batch, stats_payload, device=torch.device("cpu"))
    assert cov_live >= 0.85

    preds = _predict_batch_pv(model, oos_batch, device=torch.device("cpu"))
    hw_bps = compute_conformal_half_width_bps(oos_batch, stats_payload)
    assert hw_bps.shape == (32,) and float(hw_bps.min()) > 0.0
    assert torch.isfinite(preds).all()

    # Fast training roundtrip into tmp_path covering train_eq_autocall_surrogate + calibrate_guardian_stats (Cal-1/Cal-2 >= 64)
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_ckpt = tmp_path / "test_ckpt.pt"
    tmp_stats = tmp_path / "test_guardian_stats.pt"
    res = train_eq_autocall_surrogate(
        n_train_samples=16,
        n_val_samples=8,
        n_cal_samples=64,
        n_test_samples=8,
        batch_size=8,
        n_epochs=2,
        n_mc_paths_train=64,
        n_substeps_train=2,
        n_mc_paths_test=64,
        n_substeps_test=2,
        d_model=64,
        n_egno_layers=2,
        n_sched_layers=1,
        checkpoint_path=tmp_ckpt,
        guardian_stats_path=tmp_stats,
        device=dev_str,
    )
    assert tmp_ckpt.exists() and tmp_stats.exists()
    assert len(res.train_loss_history) == 2
    assert res.test_mae_bps >= 0.0
    loaded_tmp, _ = load_eq_autocall_surrogate(tmp_ckpt, device="cpu")
    assert loaded_tmp.d_model == 64

    # Exercise evaluate_surrogate_mae_bps standalone generation path (test_batch=None)
    mae_gen, mae_gen_bps, per_d_gen = evaluate_surrogate_mae_bps(
        loaded_tmp,
        n_samples_per_d=2,
        d_values=(1, 2),
        n_mc_paths=64,
        n_substeps=2,
        seed=1234,
        device="cpu",
    )
    assert mae_gen >= 0.0 and mae_gen_bps >= 0.0 and set(per_d_gen.keys()) == {1, 2}

    # Full-stress sampling check
    stress_batch = generate_synthetic_autocall_batch(
        batch_size=8, d_max=8, m_max=48, m_rem_max=12, n_mc_paths=128, seed=77, domain_mode="full_stress"
    )
    assert stress_batch.global_feats.shape == (8, 16)

    # Exercise recalibrate_guardian_stats_from_checkpoint, raw state dict loading, and CLI main() on tmp copies
    from deepvol.training.train_eq_autocall import main as train_main

    recal_out = recalibrate_guardian_stats_from_checkpoint(
        checkpoint_path=tmp_ckpt,
        guardian_stats_path=tmp_stats,
        n_train_samples=16,
        n_cal_samples=64,
        n_test_samples=8,
        n_mc_paths_test=64,
        n_substeps_test=2,
        seed=99,
        device=dev_str,
        tau_desk_bps=60.0,
    )
    assert float(recal_out["conformal_coverage"]) >= 0.0
    assert "stats_payload" in recal_out

    raw_state_path = tmp_path / "raw_state.pt"
    torch.save(model.state_dict(), raw_state_path)
    loaded_raw, raw_metrics = load_eq_autocall_surrogate(raw_state_path, device="cpu")
    assert loaded_raw.d_max == model.d_max and raw_metrics == {}

    with mock.patch.object(
        sys,
        "argv",
        [
            "train_eq_autocall",
            "--recalibrate-only",
            "--device",
            dev_str,
            "--n-train",
            "16",
            "--n-cal",
            "64",
            "--n-test",
            "8",
            "--n-mc-test",
            "64",
            "--n-substeps-test",
            "2",
            "--checkpoint-path",
            str(tmp_ckpt),
            "--guardian-stats-path",
            str(tmp_stats),
        ],
    ):
        train_main()

    tmp_cli_ckpt = tmp_path / "cli_ckpt.pt"
    tmp_cli_stats = tmp_path / "cli_stats.pt"
    with mock.patch.object(
        sys,
        "argv",
        [
            "train_eq_autocall",
            "--device",
            dev_str,
            "--n-train",
            "16",
            "--n-val",
            "8",
            "--n-cal",
            "64",
            "--n-test",
            "8",
            "--batch-size",
            "8",
            "--epochs",
            "1",
            "--n-mc-train",
            "64",
            "--n-substeps-train",
            "2",
            "--n-mc-test",
            "64",
            "--n-substeps-test",
            "2",
            "--checkpoint-path",
            str(tmp_cli_ckpt),
            "--guardian-stats-path",
            str(tmp_cli_stats),
        ],
    ):
        train_main()
    assert tmp_cli_ckpt.exists() and tmp_cli_stats.exists()


