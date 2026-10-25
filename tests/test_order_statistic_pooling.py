from __future__ import annotations

import pytest
import torch

from deepvol.surrogates.order_statistic_pooling import (
    OrderStatisticPooling,
    masked_log_mean_exp_max,
    masked_log_mean_exp_min,
)


def test_order_statistic_pooling_shapes_and_constant_recovery() -> None:
    torch.manual_seed(42)
    pool = OrderStatisticPooling(hidden_dim=128, out_dim=256, init_tau=0.15).to(
        dtype=torch.float64
    )
    batch_size, d_max, channels = 4, 8, 128

    base_vec = torch.randn(batch_size, 1, channels, dtype=torch.float64)
    h = base_vec.expand(batch_size, d_max, channels).clone()
    mask = torch.zeros(batch_size, d_max, dtype=torch.float64)
    for b, d_active in enumerate([1, 3, 5, 8]):
        mask[b, :d_active] = 1.0
        h[b, d_active:, :] = 9999.0

    h_mean, h_min, h_max, h_softmin, h_softmax = pool.compute_five_statistics(h, mask)
    expected = base_vec.squeeze(1)
    for stat in (h_mean, h_min, h_max, h_softmin, h_softmax):
        assert stat.shape == (batch_size, channels)
        assert torch.allclose(stat, expected, atol=1e-13, rtol=1e-13)

    out = pool(h, mask)
    assert out.shape == (batch_size, 256)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float32, 1e-6), (torch.float64, 1e-14)],
)
def test_order_statistic_pooling_sd_permutation_invariance_and_grad_equivariance(
    dtype: torch.dtype,
    tol: float,
) -> None:
    torch.manual_seed(101)
    pool = OrderStatisticPooling(hidden_dim=64, out_dim=128, init_tau=0.15).to(
        dtype=dtype
    )
    pool.eval()

    batch_size, d_max, channels = 3, 8, 64
    h = torch.randn(batch_size, d_max, channels, dtype=dtype, requires_grad=True)
    mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=dtype,
    )

    out_orig = pool(h, mask)
    grad_orig = torch.autograd.grad(out_orig.sum(), h)[0]

    perm = torch.tensor([4, 1, 0, 3, 2, 7, 5, 6], dtype=torch.long)
    inv_perm = torch.argsort(perm)

    h_perm = h.detach()[:, perm, :].clone().requires_grad_(True)
    mask_perm = mask[:, perm]

    out_perm = pool(h_perm, mask_perm)
    grad_perm = torch.autograd.grad(out_perm.sum(), h_perm)[0]
    grad_unperm = grad_perm[:, inv_perm, :]

    max_out_diff = (out_orig - out_perm).abs().max().item()
    max_grad_diff = (grad_orig - grad_unperm).abs().max().item()
    assert max_out_diff <= tol
    assert max_grad_diff <= tol


def test_order_statistic_pooling_dummy_padded_slot_exact_zero_invariance() -> None:
    torch.manual_seed(202)
    pool = OrderStatisticPooling(hidden_dim=64, out_dim=128, init_tau=0.15).eval()

    batch_size, d_max, channels = 4, 8, 64
    h_base = torch.randn(batch_size, d_max, channels, requires_grad=True)
    mask = torch.zeros(batch_size, d_max)
    mask[:, :4] = 1.0

    out_base = pool(h_base, mask)
    g1_base = torch.autograd.grad(out_base.sum(), h_base, create_graph=True)[0]
    g2_base = torch.autograd.grad(g1_base.sum(), h_base)[0]

    assert (g1_base[:, 4:, :].abs().max().item()) == 0.0
    assert (g2_base[:, 4:, :].abs().max().item()) == 0.0

    h_perturbed = h_base.detach().clone()
    h_perturbed[:, 4:, :] += 1e6
    h_perturbed[0, 7, :] = float("nan")
    h_perturbed.requires_grad_(True)

    out_pert = pool(h_perturbed, mask)
    g1_pert = torch.autograd.grad(out_pert.sum(), h_perturbed, create_graph=True)[0]
    g2_pert = torch.autograd.grad(g1_pert.sum(), h_perturbed)[0]

    assert (out_base - out_pert).abs().max().item() == 0.0
    assert (g1_base[:, :4, :] - g1_pert[:, :4, :]).abs().max().item() == 0.0
    assert (g2_base[:, :4, :] - g2_pert[:, :4, :]).abs().max().item() == 0.0
    assert g1_pert[:, 4:, :].abs().max().item() == 0.0
    assert g2_pert[:, 4:, :].abs().max().item() == 0.0


def test_order_statistic_pooling_c_infinity_hessian_continuity_at_exact_ties() -> None:
    torch.manual_seed(303)
    pool = OrderStatisticPooling(hidden_dim=32, out_dim=64, init_tau=0.15).to(
        dtype=torch.float64
    )
    pool.eval()

    d_max, channels = 6, 32
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    base_h = torch.randn(1, d_max, channels, dtype=torch.float64)
    base_h[:, 1, :] = base_h[:, 0, :].clone()

    def eval_grad_and_hessian_diag(eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = base_h.clone()
        h_in[:, 0, :] = h_in[:, 0, :] + 0.5 * eps
        h_in[:, 1, :] = h_in[:, 1, :] - 0.5 * eps
        h_in.requires_grad_(True)
        out = pool(h_in, mask)
        scalar = (out**2).sum()
        g1 = torch.autograd.grad(scalar, h_in, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.sum(), h_in)[0]
        return g1.detach(), g2.detach()

    g1_neg, g2_neg = eval_grad_and_hessian_diag(-1e-6)
    g1_tie, g2_tie = eval_grad_and_hessian_diag(0.0)
    g1_pos, g2_pos = eval_grad_and_hessian_diag(1e-6)

    assert torch.isfinite(g1_tie).all()
    assert torch.isfinite(g2_tie).all()
    assert torch.allclose(g1_tie[:, 0, :], g1_tie[:, 1, :], atol=1e-14)
    assert torch.allclose(g2_tie[:, 0, :], g2_tie[:, 1, :], atol=1e-14)
    assert (g1_pos - g1_neg).abs().max().item() < 1e-4
    assert (g2_pos - g2_neg).abs().max().item() < 1e-3
    assert (g2_tie - 0.5 * (g2_pos + g2_neg)).abs().max().item() < 1e-7


def test_w5_softmin_partition_of_unity_and_negative_semidefinite_hessian() -> None:
    torch.manual_seed(404)
    d_active = 5
    h = torch.randn(1, d_active, 1, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, d_active, dtype=torch.float64)
    tau = 0.12

    smin = masked_log_mean_exp_min(h, mask, tau).squeeze()
    smax = masked_log_mean_exp_max(h, mask, tau).squeeze()

    min_exact = h.min().item()
    max_exact = h.max().item()
    assert min_exact <= smin.item() <= max_exact
    assert min_exact <= smax.item() <= max_exact

    grad_min = torch.autograd.grad(smin, h, create_graph=True)[0].view(-1)
    assert torch.allclose(
        grad_min.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-14
    )
    assert (grad_min >= 0.0).all()

    hess_min_rows = []
    for i in range(d_active):
        row = torch.autograd.grad(grad_min[i], h, retain_graph=True)[0].view(-1)
        hess_min_rows.append(row)
    hess_min = torch.stack(hess_min_rows, dim=0)

    eigvals = torch.linalg.eigvalsh(hess_min)
    assert eigvals.max().item() <= 1e-13
    assert abs(eigvals.max().item()) <= 1e-13
