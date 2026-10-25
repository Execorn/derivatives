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
    assert batch.node_feats.shape == (16, 8, 14)
    assert batch.edge_feats.shape == (16, 8, 8, 4)
    assert batch.asset_mask.shape == (16, 8)
    assert batch.sched_feats.shape == (16, 48, 12)
    assert batch.sched_mask.shape == (16, 48)
    assert batch.global_feats.shape == (16, 10)

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
    glob_zero = torch.zeros((4, 10), dtype=torch.float32)
    w_zero = criterion.spread_attenuation_weight(glob_zero)
    assert torch.allclose(w_zero, torch.zeros_like(w_zero))

    glob_005 = torch.zeros((4, 10), dtype=torch.float32)
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
    """Gate G1.5: Verify saved checkpoint `artifacts/checkpoints/eq_autocall_egno.pt` achieves MAE <= 15.0 bps."""
    if not DEFAULT_CHECKPOINT_PATH.exists():
        train_eq_autocall_surrogate(checkpoint_path=DEFAULT_CHECKPOINT_PATH)

    assert DEFAULT_CHECKPOINT_PATH.exists(), f"Missing checkpoint at {DEFAULT_CHECKPOINT_PATH}"
    model, metrics = load_eq_autocall_surrogate(DEFAULT_CHECKPOINT_PATH, device="cpu")
    assert isinstance(model, MaskedEQAutocallSurrogate)

    # Evaluate out-of-sample test MAE across all basket sizes D in [1, 8]
    overall_mae, overall_mae_bps, per_d_mae_bps = evaluate_surrogate_mae_bps(
        model,
        n_samples_per_d=8,
        d_values=(1, 2, 3, 4, 5, 6, 7, 8),
        n_mc_paths=512,
        seed=9999,
        device="cpu",
    )
    assert overall_mae <= 0.0015, (
        f"Gate G1.5 violation: out-of-sample MAE = {overall_mae:.6f} ({overall_mae_bps:.2f} bps) > 15.0 bps"
    )
    assert overall_mae_bps <= 15.0
    for d_val, d_bps in per_d_mae_bps.items():
        assert d_bps <= 25.0, f"Per-D MAE for D={d_val} too high: {d_bps:.2f} bps"

    # Also verify a fast training roundtrip into tmp_path
    tmp_ckpt = tmp_path / "test_ckpt.pt"
    res = train_eq_autocall_surrogate(
        n_train_samples=16,
        batch_size=8,
        n_epochs=2,
        n_mc_paths=128,
        d_model=64,
        n_egno_layers=2,
        n_sched_layers=1,
        checkpoint_path=tmp_ckpt,
        eval_samples_per_d=2,
    )
    assert tmp_ckpt.exists()
    assert len(res.train_loss_history) == 2
    assert res.test_mae_bps <= 15.0
    loaded_tmp, _ = load_eq_autocall_surrogate(tmp_ckpt)
    assert loaded_tmp.d_model == 64
