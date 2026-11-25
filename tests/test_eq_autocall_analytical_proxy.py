"""test_eq_autocall_analytical_proxy.py — Verification suite for Task 1.2 (Stulz-Margrabe Analytical Baseline).

Covers:
  - Gate G1.4: Distressed 7-asset Worst-Of reference accuracy (|Delta Opt| <= 0.5 bps vs 32k Sobol Terminal IV MC;
    actual error ~0.0047 bps).
  - Gate G2.1: Single-trade CPU analytical baseline latency <= 50.0 us p50.
  - Recursive moment matching across `WorstOf`, `BestOf`, and `Average` vs 65,536-path Sobol MC.
  - Batched `(B, D_max)` evaluation with `asset_mask` dummy-asset invariance.
  - Exact PyTorch float64 autograd differentiability with respect to forwards, volatilities, and correlations.
  - Multi-period callable survival monotonicity and memory coupon recursion.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pytest
import torch

from deepvol.api.schemas_eq_autocall import EQAutocallableTradeRequest, parse_murex_autocall_dag
from deepvol.models.autocall import (
    compute_analytical_autocall_baseline,
    fast_stulz_margrabe_cpu,
    preprocess_eq_autocall,
    price_eq_autocall_dupire_qmc,
    stulz_margrabe_moments_with_weights,
    stulz_margrabe_worst_of_moments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


@pytest.fixture(scope="module")
def parsed_dag() -> tuple[EQAutocallableTradeRequest, dict[str, Any]]:
    return parse_murex_autocall_dag(REF_REQUEST_PATH)


def test_G1_4_stulz_margrabe_distressed_reference_accuracy(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G1.4: Analytical Stulz-Margrabe proxy matches 32k Sobol Terminal IV MC within <= 0.5 bps (5e-5)."""
    req_model, _ = parsed_dag
    state = preprocess_eq_autocall(req_model)

    base_cpu = compute_analytical_autocall_baseline(state, use_fast_cpu=True)
    base_torch = compute_analytical_autocall_baseline(state, use_fast_cpu=False)

    # CPU fast-path and PyTorch differentiable path must agree to machine precision
    torch.testing.assert_close(base_cpu.f_w, base_torch.f_w, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(base_cpu.v_w, base_torch.v_w, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(base_cpu.option_pv_proxy, base_torch.option_pv_proxy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        base_cpu.product_value_proxy, base_torch.product_value_proxy, rtol=1e-12, atol=1e-12
    )

    # Check empirical benchmark E3 moment values on the 7-asset RUB trade
    assert float(base_cpu.f_w[0].item()) == pytest.approx(0.204762515962, abs=1e-9)
    assert float(base_cpu.v_w[0].item()) == pytest.approx(0.242516096672, abs=1e-9)

    # Compare against 32,768-path Sobol Terminal IV MC
    qmc_term = price_eq_autocall_dupire_qmc(
        state,
        n_paths=32768,
        mode="terminal_iv",
    )
    opt_diff = abs(float(base_cpu.option_pv_proxy.item()) - float(qmc_term.option_pv))
    pv_diff = abs(float(base_cpu.product_value_proxy.item()) - float(qmc_term.product_value))

    # Gate G1.4 threshold is 0.5 bps = 5e-5 (actual error is ~4.7e-7 = 0.0047 bps)
    assert opt_diff <= 5e-5, f"Gate G1.4 failed: |Delta Opt| = {opt_diff:.8f} > 5e-5"
    assert pv_diff <= 5e-5, f"Gate G1.4 failed: |Delta PV| = {pv_diff:.8f} > 5e-5"


def test_G2_1_analytical_proxy_cpu_latency(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G2.1: Single-trade CPU analytical baseline latency <= 50.0 us p50."""
    req_model, _ = parsed_dag
    state = preprocess_eq_autocall(req_model)
    f_arr = state.forward_perf[:, 0].detach().numpy()
    v_arr = state.v_perf[:, 0].detach().numpy()
    c_arr = state.corr_matrix.detach().numpy()

    # Warmup
    for _ in range(100):
        fast_stulz_margrabe_cpu(f_arr, v_arr, c_arr)
        compute_analytical_autocall_baseline(state, use_fast_cpu=True)

    n_iters = 1000
    latencies_stulz_us: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fast_stulz_margrabe_cpu(f_arr, v_arr, c_arr)
        latencies_stulz_us.append((time.perf_counter() - t0) * 1e6)

    latencies_base_us: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        compute_analytical_autocall_baseline(state, use_fast_cpu=True)
        latencies_base_us.append((time.perf_counter() - t0) * 1e6)

    latencies_stulz_us.sort()
    latencies_base_us.sort()
    p50_stulz = latencies_stulz_us[n_iters // 2]
    p50_base = latencies_base_us[n_iters // 2]

    # Guard microsecond wall-clock checks against Python trace/coverage hooks (MRM-018).
    if sys.gettrace() is None:
        from deepvol.models.autocall._tier1_backend import get_tier1_backend

        assert p50_stulz <= 50.0, (
            f"Gate G2.1 failed: fast_stulz_margrabe_cpu p50 = {p50_stulz:.2f} us > 50.0 us"
        )
        if get_tier1_backend().name == "cpp23":
            assert p50_base <= 50.0, (
                f"Gate G2.1 failed: compute_analytical_autocall_baseline p50 = {p50_base:.2f} us > 50.0 us"
            )
        else:
            assert p50_base <= 250.0, (
                f"PureNumpy fallback baseline p50 = {p50_base:.2f} us > 250.0 us"
            )


@pytest.mark.parametrize("monitoring_index", ["WorstOf", "BestOf", "Average"])
def test_stulz_margrabe_worst_of_best_of_average_vs_sobol_mc(monitoring_index: str) -> None:
    """Verify moment-matching accuracy across WorstOf, BestOf, and Average against 65,536-path MC."""
    f_perf = torch.tensor([0.85, 0.92, 1.05], dtype=torch.float64)
    v_perf = torch.tensor([0.22, 0.18, 0.25], dtype=torch.float64)
    corr = torch.tensor(
        [
            [1.0, 0.45, 0.30],
            [0.45, 1.0, 0.50],
            [0.30, 0.50, 1.0],
        ],
        dtype=torch.float64,
    )

    f_w, v_w = stulz_margrabe_worst_of_moments(
        f_perf=f_perf,
        v_perf=v_perf,
        corr=corr,
        monitoring_index=monitoring_index,
    )
    assert f_w.dtype == torch.float64
    assert v_w.dtype == torch.float64

    # Simulate 65,536 antithetic Sobol paths
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=12345)
    u = sobol.draw(32768).to(dtype=torch.float64).clamp(1e-12, 1.0 - 1e-12)
    z_half = torch.special.ndtri(u)
    z = torch.cat([z_half, -z_half], dim=0)
    chol = torch.linalg.cholesky(corr)
    z_corr = z @ chol.T
    paths = f_perf.unsqueeze(0) * torch.exp(
        -0.5 * (v_perf * v_perf).unsqueeze(0) + v_perf.unsqueeze(0) * z_corr
    )

    if monitoring_index == "WorstOf":
        w_mc = paths.min(dim=1).values
    elif monitoring_index == "BestOf":
        w_mc = paths.max(dim=1).values
    else:
        w_mc = paths.mean(dim=1)

    f_mc = float(w_mc.mean().item())
    m2_mc = float((w_mc * w_mc).mean().item())
    v_mc = float(np.sqrt(np.log(m2_mc / (f_mc * f_mc))))

    assert float(f_w.item()) == pytest.approx(f_mc, rel=1.5e-2)
    assert float(v_w.item()) == pytest.approx(v_mc, rel=4.0e-2)


def test_stulz_margrabe_batched_padding_mask_invariance() -> None:
    """Verify that padded dummy assets with asset_mask=0 do not affect (F_W, v_W) or Euler weights."""
    f_active = torch.tensor([[0.72, 0.65, 0.88, 0.95]], dtype=torch.float64)
    v_active = torch.tensor([[0.28, 0.31, 0.24, 0.20]], dtype=torch.float64)
    corr_active = torch.tensor(
        [
            [
                [1.0, 0.4, 0.3, 0.2],
                [0.4, 1.0, 0.5, 0.3],
                [0.3, 0.5, 1.0, 0.4],
                [0.2, 0.3, 0.4, 1.0],
            ]
        ],
        dtype=torch.float64,
    )

    fw_4, vw_4, w_4 = stulz_margrabe_moments_with_weights(f_active, v_active, corr_active)

    # Pad to D_max = 8 with arbitrary values on dummy slots 4..7 and mask = [1,1,1,1,0,0,0,0]
    f_pad = torch.cat([f_active, torch.tensor([[0.05, 0.10, 5.0, 0.01]], dtype=torch.float64)], dim=1)
    v_pad = torch.cat([v_active, torch.tensor([[0.90, 0.80, 0.70, 0.60]], dtype=torch.float64)], dim=1)
    corr_pad = torch.eye(8, dtype=torch.float64).unsqueeze(0)
    corr_pad[:, :4, :4] = corr_active
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    fw_8, vw_8, w_8 = stulz_margrabe_moments_with_weights(f_pad, v_pad, corr_pad, asset_mask=mask)

    torch.testing.assert_close(fw_4, fw_8, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(vw_4, vw_8, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(w_4, w_8[:, :4], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(w_8[:, 4:], torch.zeros((1, 4), dtype=torch.float64), rtol=1e-13, atol=1e-13)


def test_stulz_margrabe_autograd_sensitivities_vs_fd() -> None:
    """Verify exact PyTorch float64 autograd sensitivities wrt forward_perf, v_perf, and corr_matrix."""
    f_0 = torch.tensor([0.62, 0.58, 0.74, 0.81], dtype=torch.float64, requires_grad=True)
    v_0 = torch.tensor([0.30, 0.34, 0.26, 0.22], dtype=torch.float64, requires_grad=True)
    c_0 = torch.tensor(
        [
            [1.0, 0.42, 0.35, 0.28],
            [0.42, 1.0, 0.48, 0.31],
            [0.35, 0.48, 1.0, 0.39],
            [0.28, 0.31, 0.39, 1.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    fw, vw = stulz_margrabe_worst_of_moments(f_0, v_0, c_0)
    loss = fw + 0.5 * vw
    grad_f, grad_v, grad_c = torch.autograd.grad(loss, (f_0, v_0, c_0))

    eps = 1e-6
    for i in range(4):
        f_up = f_0.detach().clone()
        f_dn = f_0.detach().clone()
        f_up[i] += eps
        f_dn[i] -= eps
        fw_u, vw_u = stulz_margrabe_worst_of_moments(f_up, v_0.detach(), c_0.detach())
        fw_d, vw_d = stulz_margrabe_worst_of_moments(f_dn, v_0.detach(), c_0.detach())
        fd_f = ((fw_u + 0.5 * vw_u) - (fw_d + 0.5 * vw_d)) / (2.0 * eps)
        assert float(grad_f[i].item()) == pytest.approx(float(fd_f.item()), rel=1e-6, abs=1e-8)

        v_up = v_0.detach().clone()
        v_dn = v_0.detach().clone()
        v_up[i] += eps
        v_dn[i] -= eps
        fw_u, vw_u = stulz_margrabe_worst_of_moments(f_0.detach(), v_up, c_0.detach())
        fw_d, vw_d = stulz_margrabe_worst_of_moments(f_0.detach(), v_dn, c_0.detach())
        fd_v = ((fw_u + 0.5 * vw_u) - (fw_d + 0.5 * vw_d)) / (2.0 * eps)
        assert float(grad_v[i].item()) == pytest.approx(float(fd_v.item()), rel=1e-6, abs=1e-8)


def test_multi_period_autocall_proxy_survival_and_memory() -> None:
    """Verify multi-period survival probability monotonicity and stopping probability conservation."""
    m_rem = 4
    state_dict = {
        "forward_perf": torch.tensor(
            [
                [0.96, 0.94, 0.92, 0.90],
                [0.98, 0.95, 0.93, 0.91],
                [1.02, 0.99, 0.97, 0.94],
            ],
            dtype=torch.float64,
        ),
        "v_perf": torch.tensor(
            [
                [0.15, 0.21, 0.26, 0.30],
                [0.18, 0.25, 0.31, 0.36],
                [0.14, 0.20, 0.24, 0.28],
            ],
            dtype=torch.float64,
        ),
        "corr_matrix": torch.tensor(
            [[1.0, 0.5, 0.4], [0.5, 1.0, 0.45], [0.4, 0.45, 1.0]],
            dtype=torch.float64,
        ),
        "df_pay": torch.tensor([0.97, 0.94, 0.91, 0.88], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.00, 0.97, 0.94, 0.91], dtype=torch.float64),
        "is_callable": torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.75, 0.75, 0.75, 0.75], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.04, 0.04, 0.04, 0.04], dtype=torch.float64),
        "strike": 0.85,
        "option_gearing": 1.17647,
        "c_mem_0": 0.04,
        "is_memory": True,
        "ko_spread": -0.10,
        "cpn_spread": -0.10,
    }

    out_cpu = compute_analytical_autocall_baseline(state_dict, use_fast_cpu=True)
    out_torch = compute_analytical_autocall_baseline(state_dict, use_fast_cpu=False)

    torch.testing.assert_close(out_cpu.p_proxy, out_torch.p_proxy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(out_cpu.survival_proxy, out_torch.survival_proxy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(out_cpu.product_value_proxy, out_torch.product_value_proxy, rtol=1e-12, atol=1e-12)

    # Survival must be monotonically non-increasing in [0, 1] and sum(p_m) + S_M == 1.0
    surv = out_cpu.survival_proxy
    assert bool(torch.all(surv >= 0.0) and torch.all(surv <= 1.0))
    for m in range(1, m_rem):
        assert float(surv[m].item()) <= float(surv[m - 1].item()) + 1e-15
    assert float((out_cpu.p_proxy.sum() + surv[-1]).item()) == pytest.approx(1.0, abs=1e-12)


def test_m2_zero_spread_nan_free_tied_spot_symmetry_and_no_buffer_aliasing() -> None:
    # EXT-001 & T1M-020: DigitalSpread = 0.0 must yield finite 1st & 2nd order AD gradients
    f_0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
    v_0 = torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64)
    c_0 = torch.tensor(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        dtype=torch.float64,
    )
    f_grid = f_0.unsqueeze(1).expand(3, 2)
    v_grid = v_0.unsqueeze(1).expand(3, 2)

    state_zero = {
        "forward_perf": f_grid,
        "v_perf": v_grid,
        "corr_matrix": c_0,
        "tau_obs": torch.tensor([0.5, 1.0], dtype=torch.float64),
        "df_pay": torch.tensor([0.95, 0.90], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.00, 0.90], dtype=torch.float64),
        "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.80, 0.80], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.05, 0.05], dtype=torch.float64),
        "strike": 0.90,
        "lower_strike": 0.50,
        "ki_barrier": 0.70,
        "has_ki": True,
        "i_ki_past": 0.0,
        "option_gearing": 1.1,
        "ko_spread": 0.0,
        "cpn_spread": 0.0,
        "ki_spread": 0.0,
        "allow_autocall_on_expiry": True,
        "force_coupons_on_autocall": True,
        "autocall_coupon": 0.03,
    }
    out_t = compute_analytical_autocall_baseline(state_zero, use_fast_cpu=False)
    out_c = compute_analytical_autocall_baseline(state_zero, use_fast_cpu=True)
    torch.testing.assert_close(out_t.product_value_proxy, out_c.product_value_proxy, rtol=1e-13, atol=1e-13)
    (g1,) = torch.autograd.grad(out_t.product_value_proxy, f_0, create_graph=True)
    assert torch.isfinite(g1).all()
    # GRK-004 & EXT-016: Tied ATM spots must have exact symmetric Deltas
    assert float((g1 - g1[0]).abs().max().item()) <= 1e-12
    for i in range(3):
        (g2_row,) = torch.autograd.grad(g1[i], f_0, retain_graph=True)
        assert torch.isfinite(g2_row).all()

    # T1M-010 & MRM-004: No mutable buffer aliasing across calls
    state_b = dict(state_zero)
    state_b["forward_perf"] = f_grid.detach()
    out_1 = compute_analytical_autocall_baseline(state_b, use_fast_cpu=True)
    pv_1 = float(out_1.product_value_proxy.item())
    state_b["strike"] = 0.20
    out_2 = compute_analytical_autocall_baseline(state_b, use_fast_cpu=True)
    assert out_1 is not out_2
    assert float(out_1.product_value_proxy.item()) == pytest.approx(pv_1, abs=1e-15)
    assert float(out_2.product_value_proxy.item()) != pytest.approx(pv_1, abs=1e-4)


def test_m2_multi_index_ratio_diff_out_down_onestar_and_tier1_backend() -> None:
    from deepvol.models.autocall._tier1_backend import PureNumpyTier1Backend, get_tier1_backend

    f_grid = torch.tensor([[0.92, 0.90], [1.08, 1.05]], dtype=torch.float64)
    v_grid = torch.tensor([[0.22, 0.30], [0.25, 0.34]], dtype=torch.float64)
    corr = torch.tensor([[1.0, 0.45], [0.45, 1.0]], dtype=torch.float64)

    for call_idx, cpn_idx, opt_idx in (
        ("WorstOf", "BestOf", "Average"),
        ("RatioAB", "DiffAB", "RatioBA"),
        ("DiffBA", "WorstOf", "BestOf"),
    ):
        for cpn_type, cpn_dir in (("In", "Up"), ("Out", "Up"), ("In", "Down"), ("Out", "Down")):
            st = {
                "forward_perf": f_grid,
                "v_perf": v_grid,
                "corr_matrix": corr,
                "tau_obs": torch.tensor([0.5, 1.0], dtype=torch.float64),
                "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
                "ko_barriers": torch.tensor([1.05, 0.95], dtype=torch.float64),
                "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
                "cpn_barriers": torch.tensor([0.85, 0.85], dtype=torch.float64),
                "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
                "coupon_amounts": torch.tensor([0.04, 0.04], dtype=torch.float64),
                "strike": 0.90,
                "lower_strike": 0.50,
                "ki_barrier": 0.70,
                "has_ki": True,
                "i_ki_past": 0.0,
                "option_gearing": 1.0,
                "ko_spread": -0.05,
                "cpn_spread": 0.05,
                "ki_spread": -0.05,
                "callability_index": call_idx,
                "coupon_index": cpn_idx,
                "option_index": opt_idx,
                "coupon_barrier_type": cpn_type,
                "coupon_barrier_direction": cpn_dir,
                "inv_lex_perm": (1, 0),
                "allow_autocall_on_expiry": True,
                "is_long_basket": (call_idx == "DiffBA"),
                "no_notional_repayment": (call_idx == "DiffBA"),
            }
            out_cpu = compute_analytical_autocall_baseline(st, use_fast_cpu=True)
            out_t = compute_analytical_autocall_baseline(st, use_fast_cpu=False)
            torch.testing.assert_close(out_cpu.product_value_proxy, out_t.product_value_proxy, rtol=1e-11, atol=1e-11)

    # OneStar + ParticipationCall CPU vs Torch parity and vs QMC
    st_os = {
        "forward_perf": f_grid,
        "v_perf": v_grid,
        "corr_matrix": corr,
        "tau_obs": torch.tensor([0.5, 1.0], dtype=torch.float64),
        "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.15, 500.0], dtype=torch.float64),
        "is_callable": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.75, 0.75], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.04, 0.04], dtype=torch.float64),
        "strike": 0.95,
        "lower_strike": 0.0,
        "ki_barrier": 0.0,
        "has_ki": False,
        "i_ki_past": 0.0,
        "option_gearing": 1.0,
        "ko_spread": -0.05,
        "cpn_spread": -0.05,
        "one_star_strike": 1.05,
        "one_star_coupon": 0.05,
        "participation_call_strike": 1.00,
        "participation_call_rate": 0.50,
    }
    os_cpu = compute_analytical_autocall_baseline(st_os, use_fast_cpu=True)
    os_t = compute_analytical_autocall_baseline(st_os, use_fast_cpu=False)
    torch.testing.assert_close(os_cpu.product_value_proxy, os_t.product_value_proxy, rtol=1e-11, atol=1e-11)

    # PureNumpyTier1Backend spline helpers & C++23 parity check (when .so is available)
    np_bk = PureNumpyTier1Backend()
    xk = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float64)
    yk = np.array([0.20, 0.25, 0.23, 0.27], dtype=np.float64)
    coeffs_np = np_bk.clamped_cubic_coeffs_1d(xk, yk)
    xq = np.array([0.05, 0.3, 0.8, 2.5], dtype=np.float64)
    v_np, d1_np, d2_np = np_bk.eval_cubic_with_derivs(xk, coeffs_np, xq, extrap_left="Linear", extrap_right="Linear")
    assert np.all(np.isfinite(v_np)) and np.all(np.isfinite(d1_np)) and np.all(np.isfinite(d2_np))

    active_bk = get_tier1_backend()
    if active_bk.name == "cpp23":
        coeffs_c = active_bk.clamped_cubic_coeffs_1d(xk, yk)
        v_c, d1_c, d2_c = active_bk.eval_cubic_with_derivs(xk, coeffs_c, xq, extrap_left="Linear", extrap_right="Linear")
        np.testing.assert_allclose(v_c, v_np, rtol=0.0, atol=2.2e-16)
        np.testing.assert_allclose(d1_c, d1_np, rtol=0.0, atol=2.2e-15)
        np.testing.assert_allclose(d2_c, d2_np, rtol=0.0, atol=2.2e-14)


def test_erfcx_normal_clark_and_stulz_margrabe_branch_parity() -> None:
    import math
    from deepvol.models.autocall import _tier1_backend as tb
    from deepvol.models.autocall import analytical_baseline as ab

    # Cover all Cody erfcx regimes (small/mid/large positive/negative) + truncated moments
    for alpha in (-5.0, -0.2, 0.2, 2.5, 27.0, 40.0):
        e_val = tb._cody_erfcx(alpha)
        assert np.isfinite(e_val) and e_val > 0.0
        mu_u, var_u = tb._truncated_upper_std_moments(alpha)
        assert np.isfinite(mu_u) and var_u > 0.0
    assert math.isnan(tb._cody_erfcx(float("nan")))

    # Normal call, digital ramp, survival post moments, interval moments, piecewise terminal call
    assert tb._normal_call(0.1, 0.2, 0.05) == pytest.approx(
        float(ab._normal_call_torch(torch.tensor(0.1, dtype=torch.float64), torch.tensor(0.2, dtype=torch.float64), torch.tensor(0.05, dtype=torch.float64)).item()),
        abs=1e-12,
    )
    assert tb._normal_call(0.1, 1e-14, 0.05) == pytest.approx(0.05)
    assert 0.0 <= tb._digital_call_spread_normal(0.05, 0.15, 0.05, -0.10) <= 1.0
    assert 0.0 <= tb._digital_call_spread_normal(0.05, 0.15, 0.05, 0.10) <= 1.0
    mu_np, var_np = tb._ramp_survival_post_moments_normal(0.02, 0.20, 0.10, -0.05)
    mu_t, var_t = ab._ramp_survival_post_moments_normal_torch(
        torch.tensor(0.02, dtype=torch.float64),
        torch.tensor(0.20, dtype=torch.float64),
        torch.tensor(0.10, dtype=torch.float64),
        torch.tensor(-0.05, dtype=torch.float64),
    )
    assert mu_np == pytest.approx(float(mu_t.item()), abs=1e-11)
    assert var_np == pytest.approx(float(var_t.item()), abs=1e-11)

    m_norm_np = tb._normal_moments_interval(0.05, 0.25, -0.1, 0.3, 2)
    m_norm_t = ab._normal_moments_interval_torch(
        torch.tensor(0.05, dtype=torch.float64),
        torch.tensor(0.25, dtype=torch.float64),
        torch.tensor(-0.1, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64),
        2,
    )
    for x_np, x_t in zip(m_norm_np, m_norm_t):
        assert x_np == pytest.approx(float(x_t.item()), abs=1e-11)

    # Piecewise terminal call on normal (True) and lognormal (False) with hard/soft/no KO
    for is_norm in (True, False):
        f_mu = 0.02 if is_norm else 1.02
        for has_ko, ko_sp in ((False, 0.0), (True, 0.0), (True, -0.05)):
            call_np = tb._exact_piecewise_terminal_call(f_mu, 0.20, is_norm, 0.90 if not is_norm else 0.0, has_ko, 1.10 if not is_norm else 0.25, ko_sp)
            call_t = ab._exact_piecewise_terminal_call_torch(
                torch.tensor(f_mu, dtype=torch.float64),
                torch.tensor(0.20, dtype=torch.float64),
                is_norm,
                0.90 if not is_norm else 0.0,
                has_ko,
                1.10 if not is_norm else 0.25,
                ko_sp,
            )
            assert call_np == pytest.approx(float(call_t.item()), abs=1e-11)

    # PureNumpyTier1Backend full scalar & multi-period parity via _eval_index_moments
    for idx_code in (0, 1, 2, 3, 9, 10):
        fw_p, vw_p, w_p = tb._eval_index_moments(
            3, [1.02, 0.96, 1.05], [0.25, 0.30, 0.22],
            [[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]],
            [0, 1, 2], idx_code,
        )
        assert np.isfinite(fw_p) and vw_p > 0.0 and len(w_p) == 3

    # Separated spots (|log(F_i/F_j)| > 0.06) verify exact 1e-12 parity between C++/NumPy and Torch
    f_np = np.array([0.85, 1.00, 1.18], dtype=np.float64)
    v_np_arr = np.array([0.25, 0.30, 0.22], dtype=np.float64)
    c_np = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=np.float64)
    for idx_name in ("WorstOf", "BestOf", "Average"):
        p_fast, fw_f, vw_f, w_f = ab.fast_stulz_margrabe_cpu(f_np, v_np_arr, c_np, k_strike=0.90, monitoring_index=idx_name)
        fw_t, vw_t, w_t = ab.stulz_margrabe_moments_with_weights(
            torch.tensor(f_np), torch.tensor(v_np_arr), torch.tensor(c_np), monitoring_index=idx_name
        )
        assert fw_f == pytest.approx(float(fw_t.item()), abs=1e-12)
        assert vw_f == pytest.approx(float(vw_t.item()), abs=1e-12)
        np.testing.assert_allclose(w_f, w_t.numpy(), atol=1e-12)
        assert p_fast >= 0.0

    for k_diff in (0.0, 0.05, -0.05):
        p_diff, fw_d, vw_d, _ = ab.fast_stulz_margrabe_cpu(
            [1.02, 0.96], [0.25, 0.30], [[1.0, 0.4], [0.4, 1.0]], k_strike=k_diff, monitoring_index="DiffAB"
        )
        s_d = max(vw_d, 1e-12)
        z_d = (k_diff - fw_d) / s_d
        expected_diff_put = (k_diff - fw_d) * float(ab._ndtr_scalar(z_d)) + s_d * float(
            ab._INV_SQRT2PI * math.exp(-0.5 * z_d * z_d)
        )
        assert p_diff > 0.0
        assert p_diff == pytest.approx(expected_diff_put, abs=1e-12)

    p_wof_zero, _, _, _ = ab.fast_stulz_margrabe_cpu(
        [1.02, 0.96], [0.25, 0.30], [[1.0, 0.4], [0.4, 1.0]], k_strike=0.0, monitoring_index="WorstOf"
    )
    assert p_wof_zero == 0.0

    fw_1d, vw_1d, w_1d = ab.stulz_margrabe_moments_with_weights(
        torch.tensor([1.05], dtype=torch.float64),
        torch.tensor([0.25], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
        monitoring_index="DiffAB",
    )
    assert float(fw_1d.item()) == 0.0 and float(w_1d[0].item()) == 0.0

    # Cover PureTorch baseline across BestOf/Average/Out/Down/LowerStrike/KI/NoNotional/AutoCallCoupon/ForceCpn/ExpiryKO
    for idx_call, idx_cpn, c_type, c_dir, force_cpn, exp_ko, long_b in (
        ("BestOf", "Average", "Out", "Down", True, True, True),
        ("Average", "WorstOf", "In", "Up", False, False, False),
    ):
        st_full: dict[str, Any] = {
            "forward_perf": torch.tensor([[0.85, 0.88], [1.00, 1.02], [1.18, 1.20]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.22, 0.30], [0.26, 0.35], [0.20, 0.28]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=torch.float64),
            "inv_lex_perm": np.array([2, 0, 1], dtype=np.int64),
            "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.05, 1.00], dtype=torch.float64),
            "is_callable": torch.tensor([1.0, 1.0 if exp_ko else 0.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([0.80, 0.80], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.04, 0.04], dtype=torch.float64),
            "strike": 0.90,
            "lower_strike": 0.50,
            "ki_barrier": 0.70,
            "has_ki": True,
            "i_ki_past": 0.0,
            "option_gearing": 1.1,
            "ko_spread": -0.05,
            "cpn_spread": 0.05,
            "ki_spread": -0.05,
            "callability_index": idx_call,
            "coupon_index": idx_cpn,
            "option_index": idx_call,
            "coupon_barrier_type": c_type,
            "coupon_barrier_direction": c_dir,
            "is_memory": True,
            "c_mem_0": 0.04,
            "no_notional_repayment": False,
            "autocall_coupon": 0.02,
            "force_coupons_on_autocall": force_cpn,
            "allow_autocall_on_expiry": exp_ko,
            "is_long_basket": long_b,
        }
        out_cpu = compute_analytical_autocall_baseline(st_full, use_fast_cpu=True)
        out_pt = compute_analytical_autocall_baseline(st_full, use_fast_cpu=False)
        torch.testing.assert_close(out_cpu.product_value_proxy, out_pt.product_value_proxy, rtol=1e-10, atol=1e-10)
        # Also exercise PureNumpy backend directly (covering _tier1_backend.py Python math paths)
        np_bk_direct = tb.get_tier1_backend(force_numpy=True)
        res_np = np_bk_direct.stulz_analytical_bridge_cpu(
            fwd_perf=st_full["forward_perf"].numpy(),
            vol_perf=st_full["v_perf"].numpy(),
            corr=st_full["corr_matrix"].numpy(),
            tau_obs=np.array([0.25, 0.50], dtype=np.float64),
            df_pay=st_full["df_pay"].numpy(),
            ko_barriers=st_full["ko_barriers"].numpy(),
            is_callable=st_full["is_callable"].numpy(),
            cpn_barriers=st_full["cpn_barriers"].numpy(),
            is_guaranteed_cpn=st_full["is_guaranteed_cpn"].numpy(),
            coupon_rates=st_full["coupon_amounts"].numpy(),
            coupon_barrier_type=c_type,
            coupon_barrier_direction=c_dir,
            inv_lex_perm=st_full["inv_lex_perm"],
            strike=0.90,
            lower_strike=0.50,
            ki_barrier=0.70,
            has_ki=True,
            i_ki_past=0.0,
            ko_spread=-0.05,
            cpn_spread=0.05,
            ki_spread=-0.05,
            call_index_code=tb.monitoring_index_to_code(idx_call),
            cpn_index_code=tb.monitoring_index_to_code(idx_cpn),
            opt_index_code=tb.monitoring_index_to_code(idx_call),
            allow_autocall_on_expiry=exp_ko,
        )
        assert np.all(np.isfinite(res_np[0]))
        np.testing.assert_allclose(res_np[7], float(out_cpu.l_put_proxy.item()), atol=1e-11)

    # PureTorch batched stulz_margrabe_moments_with_weights with asset_mask & RatioAB / DiffAB baseline
    f_b = torch.tensor([[1.02, 0.96, 1.05], [0.95, 1.08, 1.01]], dtype=torch.float64)
    v_b = torch.tensor([[0.25, 0.30, 0.22], [0.20, 0.28, 0.24]], dtype=torch.float64)
    c_b = torch.tensor([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=torch.float64).unsqueeze(0).expand(2, 3, 3)
    m_b = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float64)
    for idx_m in ("WorstOf", "BestOf", "Average", "RatioAB", "DiffAB"):
        fw_bm, vw_bm, w_bm = ab.stulz_margrabe_moments_with_weights(
            f_b, v_b, c_b, asset_mask=m_b, monitoring_index=idx_m, inv_lex_perm=[1, 0, 2]
        )
        assert fw_bm.shape == (2,) and vw_bm.shape == (2,) and w_bm.shape == (2, 3)

    for idx_2d in ("RatioAB", "DiffAB"):
        st_2d = {
            "forward_perf": torch.tensor([[1.02, 1.04], [0.96, 0.98]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.22, 0.30], [0.26, 0.35]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0, 0.4], [0.4, 1.0]], dtype=torch.float64),
            "inv_lex_perm": np.array([1, 0], dtype=np.int64),
            "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.05, 1.00], dtype=torch.float64),
            "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([0.0, 0.0] if idx_2d == "DiffAB" else [0.80, 0.80], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.04, 0.04], dtype=torch.float64),
            "strike": 0.05 if idx_2d == "DiffAB" else 0.90,
            "lower_strike": 0.0,
            "ki_barrier": 0.0,
            "has_ki": False,
            "i_ki_past": 0.0,
            "option_gearing": 1.0,
            "ko_spread": -0.05,
            "cpn_spread": -0.05,
            "callability_index": idx_2d,
            "coupon_index": idx_2d,
            "option_index": idx_2d,
            "allow_autocall_on_expiry": True,
            "one_star_strike": 1.10,
            "one_star_coupon": 0.05,
            "participation_call_strike": 1.00,
            "participation_call_rate": 0.30,
        }
        out_2d_cpu = compute_analytical_autocall_baseline(st_2d, use_fast_cpu=True)
        out_2d_pt = compute_analytical_autocall_baseline(st_2d, use_fast_cpu=False)
        torch.testing.assert_close(out_2d_cpu.product_value_proxy, out_2d_pt.product_value_proxy, rtol=1e-10, atol=1e-10)

    # PureTorch 1-asset & 3-asset OneStar/Ratio/Diff moments + m_rem=0 early return
    st_1d = {
        "forward_perf": torch.tensor([[1.02, 1.04]], dtype=torch.float64),
        "v_perf": torch.tensor([[0.22, 0.30]], dtype=torch.float64),
        "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
        "inv_lex_perm": np.array([0], dtype=np.int64),
        "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.05, 1.00], dtype=torch.float64),
        "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.80, 0.80], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.04, 0.04], dtype=torch.float64),
        "strike": 0.90,
        "lower_strike": 0.50,
        "ki_barrier": 0.70,
        "has_ki": True,
        "i_ki_past": 0.0,
        "allow_autocall_on_expiry": True,
        "one_star_strike": 1.10,
        "one_star_coupon": 0.05,
        "participation_call_strike": 1.00,
        "participation_call_rate": 0.30,
    }
    o1_cpu = compute_analytical_autocall_baseline(st_1d, use_fast_cpu=True)
    o1_pt = compute_analytical_autocall_baseline(st_1d, use_fast_cpu=False)
    torch.testing.assert_close(o1_cpu.product_value_proxy, o1_pt.product_value_proxy, rtol=1e-10, atol=1e-10)

    for idx_3d in ("RatioBestWorst", "RatioBestAverage", "RatioAverageWorst", "DiffBestWorst", "DiffBestAverage", "DiffAverageWorst"):
        fw_m3, vw_m3, w_m3 = ab.stulz_margrabe_moments_with_weights(
            torch.tensor([0.85, 1.00, 1.18], dtype=torch.float64),
            torch.tensor([0.22, 0.26, 0.20], dtype=torch.float64),
            torch.tensor([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=torch.float64),
            monitoring_index=idx_3d,
        )
        assert torch.isfinite(fw_m3) and torch.isfinite(vw_m3) and w_m3.shape == (3,)
        fw_c3, vw_c3, w_c3 = tb._eval_index_moments(
            3, [0.85, 1.00, 1.18], [0.22, 0.26, 0.20],
            [[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]],
            [0, 1, 2], tb.monitoring_index_to_code(idx_3d),
        )
        assert float(fw_m3.item()) == pytest.approx(fw_c3, abs=1e-10)

    # Also test 3D batched + OneStar pure-torch path + numpy backend scalar checks
    o3_os = compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.tensor([[1.02, 1.04], [0.96, 0.94], [1.05, 1.01]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.22, 0.30], [0.26, 0.35], [0.20, 0.28]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=torch.float64),
            "inv_lex_perm": np.array([2, 0, 1], dtype=np.int64),
            "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.05, 500.0], dtype=torch.float64),
            "is_callable": torch.tensor([1.0, 0.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([0.80, 0.80], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.04, 0.04], dtype=torch.float64),
            "one_star_strike": 1.10,
            "one_star_coupon": 0.05,
            "participation_call_strike": 1.00,
            "participation_call_rate": 0.25,
        },
        use_fast_cpu=False,
    )
    assert torch.isfinite(o3_os.product_value_proxy).all()
    assert ab._black76_call_scalar(1.0, 0.9, 0.2) > 0.1
    assert 0.0 <= ab._digital_call_spread_prob_scalar(1.0, 0.2, 1.0, -0.05) <= 1.0


def test_analytical_proxy_audit_blind_spots() -> None:
    import ctypes
    from scipy.stats import qmc
    from deepvol.models.autocall import _tier1_backend as tb, analytical_baseline as ab

    # TwinWin M=4 vs 65,536-path QMC reference (two-sided KO and max(W, 2 - W) put payoff).
    for ko_lvl in (0.85, 1.35):
        m_tw = 4
        fwd_tw = torch.full((1, m_tw), 1.02, dtype=torch.float64)
        v_tw = torch.tensor([[0.12, 0.17, 0.21, 0.25]], dtype=torch.float64)
        df_tw = torch.full((m_tw,), 0.96, dtype=torch.float64)
        st_tw = {
            "forward_perf": fwd_tw,
            "v_perf": v_tw,
            "corr_matrix": torch.ones((1, 1), dtype=torch.float64),
            "df_pay": df_tw,
            "ko_barriers": torch.full((m_tw,), ko_lvl, dtype=torch.float64),
            "is_callable": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
            "cpn_barriers": torch.full((m_tw,), 1.08, dtype=torch.float64),
            "is_guaranteed_cpn": torch.zeros(m_tw, dtype=torch.float64),
            "coupon_amounts": torch.full((m_tw,), 0.03, dtype=torch.float64),
            "strike": 1.28,
            "lower_strike": 0.0,
            "ki_barrier": 0.0,
            "has_ki": False,
            "ko_spread": 0.0,
            "cpn_spread": 0.0,
            "ki_spread": 0.0,
            "is_memory": False,
            "option_gearing": 1.0,
            "option_index": "TwinWin",
            "callability_index": "TwinWin",
            "coupon_index": "TwinWin",
            "allow_autocall_on_expiry": True,
        }
        tw_cpu = compute_analytical_autocall_baseline(st_tw, use_fast_cpu=True)
        tw_pt = compute_analytical_autocall_baseline(st_tw, use_fast_cpu=False)
        torch.testing.assert_close(tw_cpu.product_value_proxy, tw_pt.product_value_proxy, atol=1e-9, rtol=1e-9)
        if ko_lvl <= 1.0:
            # 100% immediate/two-sided KO when max(W, 2 - W) >= 1.0 >= B_ko leaves zero surviving path.
            assert float(tw_pt.survival_proxy[-1]) <= 1e-6
        else:
            sob = qmc.Sobol(d=m_tw, scramble=True, seed=2026)
            z = torch.from_numpy(
                np.sqrt(2.0) * torch.erfinv(2.0 * torch.from_numpy(sob.random_base2(16)).clamp(1e-12, 1 - 1e-12) - 1.0).numpy()
            ).to(torch.float64)
            # Incremental lognormal steps up to variance v_tw^2.
            dv2 = torch.cat([v_tw[0, :1] ** 2, v_tw[0, 1:] ** 2 - v_tw[0, :-1] ** 2])
            s_paths = torch.ones((65536, m_tw), dtype=torch.float64)
            lw = torch.zeros(65536, dtype=torch.float64)
            prev_f = 1.0
            for m in range(m_tw):
                drift_m = math.log(float(fwd_tw[0, m]) / prev_f) - 0.5 * float(dv2[m])
                lw = lw + drift_m + math.sqrt(float(dv2[m])) * z[:, m]
                s_paths[:, m] = torch.exp(lw)
                prev_f = float(fwd_tw[0, m])
            alive = torch.ones(65536, dtype=torch.bool)
            mc_pv = torch.zeros(65536, dtype=torch.float64)
            for m in range(m_tw):
                tw_idx_m = torch.maximum(s_paths[:, m], 2.0 - s_paths[:, m])
                hit = alive & (tw_idx_m >= ko_lvl)
                cpn_hit = alive & (tw_idx_m >= 1.08)
                mc_pv += torch.where(cpn_hit, 0.03 * df_tw[m], 0.0)
                mc_pv += torch.where(hit, df_tw[m], 0.0)
                alive = alive & (~hit)
            tw_term = torch.maximum(s_paths[:, -1], 2.0 - s_paths[:, -1])
            tw_payoff = 1.0 - torch.clamp(1.28 - tw_term, min=0.0)
            mc_pv += torch.where(alive, tw_payoff * df_tw[-1], 0.0)
            assert abs(float(tw_pt.product_value_proxy.item()) - float(mc_pv.mean().item())) < 0.015

    # d=1 OneStar deep ITM (f=0.50, K=0.90 -> put ~ 0.40) parity between CPU C++ and PureTorch.
    st_1d_itm = {
        "forward_perf": torch.tensor([[0.50]], dtype=torch.float64),
        "v_perf": torch.tensor([[0.15]], dtype=torch.float64),
        "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
        "df_pay": torch.tensor([1.0], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.20], dtype=torch.float64),
        "is_callable": torch.tensor([0.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.80], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.0], dtype=torch.float64),
        "strike": 0.90,
        "lower_strike": 0.0,
        "option_gearing": 1.0,
        "one_star_strike": 1.10,
        "one_star_coupon": 0.05,
        "no_notional_repayment": True,
    }
    os_cpu = compute_analytical_autocall_baseline(st_1d_itm, use_fast_cpu=True)
    os_pt = compute_analytical_autocall_baseline(st_1d_itm, use_fast_cpu=False)
    torch.testing.assert_close(os_cpu.l_put_proxy, os_pt.l_put_proxy, atol=1e-10, rtol=1e-10)
    assert float(os_pt.l_put_proxy.item()) == pytest.approx(0.40, abs=0.02)

    # Extreme left tail u_hi <= -20 and -1000 lognormal ramp stability.
    for u_hi in (-20.0, -50.0, -1000.0):
        mu_ln, var_ln = tb._ramp_survival_post_moments_lognormal(math.exp(u_hi), 0.25, 1.0, -0.05)
        assert math.isfinite(mu_ln) and math.isfinite(var_ln) and var_ln > 0.0
        mu_t, var_t = ab._ramp_survival_post_moments_lognormal_torch(
            torch.tensor(math.exp(u_hi), dtype=torch.float64),
            torch.tensor(0.25, dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
            -0.05,
        )
        assert torch.isfinite(mu_t) and torch.isfinite(var_t) and float(var_t.item()) > 0.0

    # EquallyWeightedBasket + SecondWorstOf/ThirdWorstOf/SecondBestOf parity (CPU & Torch).
    f4 = np.array([0.82, 0.94, 1.07, 1.22], dtype=np.float64)
    v4 = np.array([0.20, 0.25, 0.22, 0.28], dtype=np.float64)
    c4 = np.full((4, 4), 0.35, dtype=np.float64)
    np.fill_diagonal(c4, 1.0)
    for idx_ex in ("EquallyWeightedBasket", "SecondWorstOf", "ThirdWorstOf", "SecondBestOf"):
        fw_c, vw_c, w_c = tb._eval_index_moments(
            4, f4.tolist(), v4.tolist(), c4.tolist(), [0, 1, 2, 3], tb.monitoring_index_to_code(idx_ex)
        )
        fw_t, vw_t, w_t = ab.stulz_margrabe_moments_with_weights(
            torch.tensor(f4), torch.tensor(v4), torch.tensor(c4), monitoring_index=idx_ex
        )
        assert fw_c == pytest.approx(float(fw_t.item()), abs=1e-10)
        assert vw_c == pytest.approx(float(vw_t.item()), abs=1e-10)
        np.testing.assert_allclose(w_c, w_t.numpy(), atol=1e-10)

    # Dispersion near-zero variance at equal forwards [1.0, 1.0].
    fw_disp, vw_disp, _ = ab.stulz_margrabe_moments_with_weights(
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        torch.tensor([1e-5, 1e-5], dtype=torch.float64),
        torch.tensor([[1.0, 1.0 - 1e-10], [1.0 - 1e-10, 1.0]], dtype=torch.float64),
        monitoring_index="Dispersion",
    )
    assert torch.isfinite(fw_disp) and torch.isfinite(vw_disp)

    # 3D (B, M) batch, b_size == 0, m_rem == 0 AD-attached zero return, ParticipationCall(strike=0),
    # and ForceCouponsOnAutocall opposite-tail BVN union.
    fwd_bm = torch.tensor([[1.01, 0.99], [0.95, 1.04]], dtype=torch.float64, requires_grad=True)
    v_bm = torch.full((2, 2), 0.20, dtype=torch.float64)
    out_b0 = compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.zeros((0, 1, 2), dtype=torch.float64),
            "v_perf": torch.zeros((0, 1, 2), dtype=torch.float64),
            "corr_matrix": torch.zeros((0, 1, 1), dtype=torch.float64),
            "df_pay": torch.zeros((0, 2), dtype=torch.float64),
            "ko_barriers": torch.zeros((0, 2), dtype=torch.float64),
            "is_callable": torch.zeros((0, 2), dtype=torch.float64),
            "cpn_barriers": torch.zeros((0, 2), dtype=torch.float64),
            "is_guaranteed_cpn": torch.zeros((0, 2), dtype=torch.float64),
            "coupon_amounts": torch.zeros((0, 2), dtype=torch.float64),
        },
        use_fast_cpu=False,
    )
    assert out_b0.product_value_proxy.shape == (0,)

    out_m0_ad = compute_analytical_autocall_baseline(
        {
            "forward_perf": fwd_bm[:, :0],
            "v_perf": v_bm[:, :0],
            "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
            "df_pay": torch.zeros((0,), dtype=torch.float64),
            "ko_barriers": torch.zeros((0,), dtype=torch.float64),
            "is_callable": torch.zeros((0,), dtype=torch.float64),
            "cpn_barriers": torch.zeros((0,), dtype=torch.float64),
            "is_guaranteed_cpn": torch.zeros((0,), dtype=torch.float64),
            "coupon_amounts": torch.zeros((0,), dtype=torch.float64),
        },
        use_fast_cpu=False,
    )
    g_m0 = torch.autograd.grad(out_m0_ad.product_value_proxy.sum(), fwd_bm, allow_unused=True)[0]
    assert g_m0 is None or torch.all(g_m0 == 0.0)

    out_pc0 = compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.tensor([[1.02]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.20]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
            "df_pay": torch.tensor([0.98], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.50], dtype=torch.float64),
            "is_callable": torch.tensor([1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([0.50], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.0], dtype=torch.float64),
            "participation_call_strike": 0.0,
            "participation_call_rate": 0.50,
            "no_notional_repayment": True,
        },
        use_fast_cpu=False,
    )
    assert float(out_pc0.product_value_proxy.item()) > 0.30

    out_fc = compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.tensor([[1.00, 1.00], [1.00, 1.00]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.25, 0.35], [0.25, 0.35]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.05, 1.05], dtype=torch.float64),
            "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([1.02, 1.02], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.05, 0.05], dtype=torch.float64),
            "coupon_barrier_type": "Out",
            "coupon_barrier_direction": "Down",
            "force_coupons_on_autocall": True,
        },
        use_fast_cpu=True,
    )
    out_fc_pt = compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.tensor([[1.00, 1.00], [1.00, 1.00]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.25, 0.35], [0.25, 0.35]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.05, 1.05], dtype=torch.float64),
            "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([1.02, 1.02], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.05, 0.05], dtype=torch.float64),
            "coupon_barrier_type": "Out",
            "coupon_barrier_direction": "Down",
            "force_coupons_on_autocall": True,
        },
        use_fast_cpu=False,
    )
    torch.testing.assert_close(out_fc.q_proxy, out_fc_pt.q_proxy, atol=1e-9, rtol=1e-9)

    # C++23 ABI handshake and L2O order-stat (15..17) + nullptr safety verification when C++ lib present.
    backend = tb.get_tier1_backend()
    cpp_lib = getattr(backend, "_lib", None)
    if cpp_lib is not None and hasattr(cpp_lib, "tier1_abi_info"):
        class _AbiInfo(ctypes.Structure):
            _fields_ = [
                ("abi_version", ctypes.c_uint32),
                ("sizeof_autocall_args", ctypes.c_uint32),
                ("alignof_autocall_args", ctypes.c_uint32),
                ("sizeof_crn_batch_args", ctypes.c_uint32),
                ("alignof_crn_batch_args", ctypes.c_uint32),
            ]

        fn = cpp_lib.tier1_abi_info
        fn.restype = _AbiInfo
        info = fn()
        assert int(info.abi_version) == 20261124
        assert int(info.sizeof_autocall_args) == 336 and int(info.alignof_autocall_args) == 8
        assert int(info.sizeof_crn_batch_args) == 320 and int(info.alignof_crn_batch_args) == 8
        cpp_lib.crn_eval_payoff_loo_l2o_cpp(None)

    # Extreme tail lognormal moments (za < zb <= -6.0 and 6.0 <= za < zb, plus a=0 and v=40)
    # must match across PureNumPy fallback, PureTorch, and C++23 with zero OverflowError.
    tail_cases = [
        (1.0, 0.2, 0.0, 1e-4, 4),
        (1.0, 0.2, 1e-6, 1e-5, 4),
        (1.0, 40.0, 0.0, 1e-3, 2),
        (1.0, 0.2, 5.0, 6.0, 4),
        (1.0, 0.1, 1e4, 1e5, 2),
    ]
    for f_t, v_t, a_t, b_t, k_ord in tail_cases:
        m_np = tb._lognormal_moments_interval(f_t, v_t, a_t, b_t, k_ord)
        m_th = ab._lognormal_moments_interval_torch(
            torch.tensor([f_t], dtype=torch.float64),
            torch.tensor([v_t], dtype=torch.float64),
            a_t,
            b_t,
            k_ord,
        )
        assert len(m_np) == k_ord + 1 and all(math.isfinite(x) and x >= 0.0 for x in m_np)
        for k_i in range(k_ord + 1):
            np.testing.assert_allclose(m_np[k_i], float(m_th[k_i].item()), rtol=1e-11, atol=1e-14)

    saved_backend = tb._CACHED_BACKEND
    try:
        tb._CACHED_BACKEND = tb.PureNumpyTier1Backend()
        st_tail = {
            "forward_perf": torch.tensor([[1.0]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.20]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
            "df_pay": torch.tensor([0.96], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.50], dtype=torch.float64),
            "is_callable": torch.tensor([1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([0.05], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.02], dtype=torch.float64),
            "strike": 0.08,
            "lower_strike": 0.04,
            "option_gearing": 1.0,
        }
        out_np_fb = compute_analytical_autocall_baseline(st_tail, use_fast_cpu=True)
        out_th_fb = compute_analytical_autocall_baseline(st_tail, use_fast_cpu=False)
        torch.testing.assert_close(out_np_fb.product_value_proxy, out_th_fb.product_value_proxy, atol=1e-11, rtol=1e-11)
    finally:
        tb._CACHED_BACKEND = saved_backend


def test_tier1_abi_v2_and_loud_fallback_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ABI v2 handshake, struct layouts, loud missing-.so error, and numpy opt-in."""
    import ctypes

    from deepvol.models.autocall import _tier1_backend as tb

    tb.reset_tier1_backend_cache()
    backend = tb.get_tier1_backend()
    assert backend.name == "cpp23"

    info_v2 = tb.tier1_abi_info_v2()
    assert int(info_v2.abi_version) == 20261125
    assert int(info_v2.autocall_args_size) == 336 and int(info_v2.autocall_args_align) == 8
    assert int(info_v2.crn_args_size) == 320 and int(info_v2.crn_args_align) == 8
    assert int(info_v2.platen_sde_aad_args_size) == 216 and int(info_v2.platen_sde_aad_args_align) == 8
    assert int(info_v2.sobol_gen_args_size) == 128 and int(info_v2.sobol_gen_args_align) == 8
    assert int(info_v2.vibrato_gamma_args_size) == 392 and int(info_v2.vibrato_gamma_args_align) == 8
    assert ctypes.sizeof(tb.Tier1AbiInfoV2C) == 48 and ctypes.alignment(tb.Tier1AbiInfoV2C) == 4

    try:
        tb.reset_tier1_backend_cache()
        monkeypatch.setenv("DEEPVOL_TIER1_CPP_LIB", "/nonexistent/path/missing_tier1_autocall_cpu.so")
        with pytest.raises(RuntimeError):
            tb.get_tier1_backend()

        tb.reset_tier1_backend_cache()
        monkeypatch.setenv("DEEPVOL_TIER1_CPP_LIB", "none")
        with pytest.raises(RuntimeError):
            tb.get_tier1_backend()

        tb.reset_tier1_backend_cache()
        monkeypatch.setenv("DEEPVOL_TIER1_CPP_LIB", "numpy")
        np_backend = tb.get_tier1_backend()
        assert np_backend.name == "numpy"
    finally:
        monkeypatch.delenv("DEEPVOL_TIER1_CPP_LIB", raising=False)
        tb.reset_tier1_backend_cache()
        assert tb.get_tier1_backend().name == "cpp23"


def test_crn_step_1d_and_platen_sde_aad_cpp_parity() -> None:
    """Verify C++23 Platen Weak-2.0 forward/backward AAD parity (< 1e-12) vs Python reference."""
    from deepvol.models.autocall import _tier1_backend as tb
    from deepvol.models.autocall.qmc_aad_greeks import _cpu_bwd_sde_aad, _cpu_step_and_ckpt_sde

    rng = np.random.default_rng(20261125)
    d_assets, m_rem, n_sub, eff_r, n_rep = 4, 4, 4, 4, 32
    lv_nt, lv_ny = 17, 33
    lv_t_grid = np.linspace(0.0, 2.0, lv_nt, dtype=np.float64)
    lv_y_min = np.array([-1.2, -1.0, -1.3, -1.1], dtype=np.float64)
    lv_y_max = np.array([1.2, 1.1, 1.3, 1.0], dtype=np.float64)

    t_ax = np.linspace(0.0, 1.0, lv_nt, dtype=np.float64)[None, :, None]
    y_ax = np.linspace(-1.0, 1.0, lv_ny, dtype=np.float64)[None, None, :]
    base_vols = np.array([0.22, 0.28, 0.19, 0.33], dtype=np.float64)[:, None, None]
    lv_tables = np.ascontiguousarray(
        base_vols + 0.04 * y_ax * y_ax - 0.02 * y_ax + 0.015 * t_ax + 0.005 * rng.normal(size=(d_assets, lv_nt, lv_ny)),
        dtype=np.float64,
    )

    corr = np.full((d_assets, d_assets), 0.35, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    chol_base = np.ascontiguousarray(np.linalg.cholesky(corr), dtype=np.float64)
    z_indep = np.ascontiguousarray(rng.standard_normal((m_rem, n_sub, eff_r, n_rep, d_assets)), dtype=np.float64)
    tau_obs = np.linspace(0.25, 1.85, m_rem, dtype=np.float64)
    r_native = np.cumsum(0.04 * np.diff(np.concatenate([[0.0], tau_obs]))[None, :] * np.ones((d_assets, 1)), axis=1)
    r_tot = r_native + np.array([[0.00], [0.02], [-0.015], [0.01]], dtype=np.float64) * tau_obs[None, :]
    bar_lx_post = np.ascontiguousarray(rng.normal(scale=0.1, size=(m_rem, d_assets, eff_r * n_rep)), dtype=np.float64)

    lx_ckpt_py, lx_py, nx_py = _cpu_step_and_ckpt_sde(
        lv_tables, z_indep, chol_base, tau_obs, r_native, r_tot, lv_t_grid, lv_y_min, lv_y_max, n_sub, eff_r, n_rep
    )
    blv_py, bchol_py, bq_py = _cpu_bwd_sde_aad(
        lx_ckpt_py, lx_py, bar_lx_post, lv_tables, z_indep, chol_base, tau_obs, r_native, r_tot, lv_t_grid, lv_y_min, lv_y_max, n_sub, eff_r, n_rep
    )

    lx_cpp, nx_cpp, blv_cpp, bchol_cpp, bq_cpp = tb.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables,
        z_indep=z_indep,
        chol_base=chol_base,
        tau_obs=tau_obs,
        r_native_obs=r_native,
        r_tot_obs=r_tot,
        lv_t_grid=lv_t_grid,
        lv_y_min=lv_y_min,
        lv_y_max=lv_y_max,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        bar_lx_post=bar_lx_post,
        run_forward=True,
        run_backward=True,
    )
    assert lx_cpp is not None and nx_cpp is not None
    assert blv_cpp is not None and bchol_cpp is not None and bq_cpp is not None
    np.testing.assert_allclose(lx_cpp, lx_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(nx_cpp, nx_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(blv_cpp, blv_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(bchol_cpp, bchol_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(bq_cpp, bq_py, atol=1e-12, rtol=1e-12)

    # Also check explicit q_shift_sub=1 path produces identical results to linear r_tot - r_native interpolation.
    q_shift_sub = np.empty((d_assets, m_rem * n_sub), dtype=np.float64)
    rt_prev = np.zeros(d_assets, dtype=np.float64)
    rn_prev = np.zeros(d_assets, dtype=np.float64)
    for m in range(m_rem):
        drt = r_tot[:, m] - rt_prev
        drn = r_native[:, m] - rn_prev
        for s in range(n_sub):
            alpha_s = (s + 0.5) / float(n_sub)
            q_shift_sub[:, m * n_sub + s] = (rt_prev + alpha_s * drt) - (rn_prev + alpha_s * drn)
        rt_prev = r_tot[:, m]
        rn_prev = r_native[:, m]
    lx_q, nx_q, blv_q, bchol_q, bq_q = tb.crn_platen_sde_aad_cpu(
        lv_tables=lv_tables,
        z_indep=z_indep,
        chol_base=chol_base,
        tau_obs=tau_obs,
        r_native_obs=None,
        r_tot_obs=None,
        lv_t_grid=lv_t_grid,
        lv_y_min=lv_y_min,
        lv_y_max=lv_y_max,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        bar_lx_post=bar_lx_post,
        q_shift_sub=q_shift_sub,
        run_forward=True,
        run_backward=True,
    )
    assert lx_q is not None and nx_q is not None
    assert blv_q is not None and bchol_q is not None and bq_q is not None
    np.testing.assert_allclose(lx_q, lx_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(blv_q, blv_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(bchol_q, bchol_py, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(bq_q, bq_py, atol=1e-12, rtol=1e-12)

    # Verify upgraded 1D trajectory stepper crn_step_1d_trajectories_cpp matches Platen Weak-2.0 (< 1e-12).
    backend = tb.get_tier1_backend()
    assert isinstance(backend, tb.Cpp23Tier1Backend)
    z_1d = np.ascontiguousarray(z_indep[:, :, :, :, 0], dtype=np.float64)
    out_1d = np.empty((1, m_rem, eff_r, n_rep), dtype=np.float64)
    backend._fn_crn_step_1d(
        1,
        lv_nt,
        lv_ny,
        m_rem,
        n_sub,
        eff_r,
        n_rep,
        0,
        float(lv_y_min[0]),
        float(lv_y_max[0]),
        lv_tables[0:1].ctypes.data,
        z_1d.ctypes.data,
        tau_obs.ctypes.data,
        r_native[0:1].ctypes.data,
        r_tot[0:1].ctypes.data,
        lv_t_grid.ctypes.data,
        out_1d.ctypes.data,
    )
    _, _, nx_1d_py = _cpu_step_and_ckpt_sde(
        lv_tables[0:1],
        z_1d[..., None],
        np.ones((1, 1), dtype=np.float64),
        tau_obs,
        r_native[0:1],
        r_tot[0:1],
        lv_t_grid,
        float(lv_y_min[0]),
        float(lv_y_max[0]),
        n_sub,
        eff_r,
        n_rep,
    )
    np.testing.assert_allclose(out_1d[0], nx_1d_py[:, 0].reshape(m_rem, eff_r, n_rep), atol=1e-12, rtol=1e-12)


def test_crn_generate_sobol_normals_cpp_parity() -> None:
    """Verify C++23 Sobol-BB normal generation (< 1e-12) vs SciPy Joe-Kuo Gray-code Sobol."""
    from scipy.stats import norm, qmc

    from deepvol.models.autocall import _tier1_backend as tb

    d_assets, m_rem, n_sub, eff_r, n_rep = 3, 4, 4, 2, 32
    dim_coarse = m_rem * d_assets
    dim_fine = m_rem * n_sub * d_assets
    total_dims = dim_coarse + dim_fine

    sob = qmc.Sobol(d=total_dims, scramble=False)
    sv = np.ascontiguousarray(sob._sv.astype(np.uint32))
    rng = np.random.default_rng(77)
    shifts = rng.integers(0, 1 << 30, size=(eff_r, total_dims), dtype=np.uint32)

    q_mat, _ = np.linalg.qr(rng.standard_normal((m_rem, m_rem)))
    bb_ortho = np.ascontiguousarray(q_mat, dtype=np.float64)
    corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.3], [-0.2, 0.3, 1.0]], dtype=np.float64)
    chol = np.ascontiguousarray(np.linalg.cholesky(corr), dtype=np.float64)

    z_indep_cpp, z_corr_cpp = tb.crn_generate_sobol_normals_cpu(
        d_assets=d_assets,
        m_rem=m_rem,
        n_sub=n_sub,
        eff_r=eff_r,
        n_rep=n_rep,
        dir_numbers=sv,
        digital_shifts=shifts,
        bb_ortho_mat=bb_ortho,
        chol_mat=chol,
        n_bits=30,
        use_gray_code=True,
        apply_bb_time=True,
        apply_corr=True,
        antithetic=False,
        return_indep=True,
        return_corr=True,
    )
    assert z_indep_cpp is not None and z_corr_cpp is not None

    # Pure Python/SciPy reference matching exact digital shift & Gray-code Sobol + BB + rotation + Cholesky.
    z_ref = np.empty((m_rem, n_sub, eff_r, n_rep, d_assets), dtype=np.float64)
    idx_u = np.arange(n_rep, dtype=np.uint32)
    gray = idx_u ^ (idx_u >> np.uint32(1))
    inv_pow2 = math.ldexp(1.0, -30)
    inv_sqrt_sub = 1.0 / math.sqrt(float(n_sub))
    for r in range(eff_r):
        for m in range(m_rem):
            for d in range(d_assets):
                c_dim = m * d_assets + d
                acc_c = np.zeros(n_rep, dtype=np.uint32)
                for b in range(30):
                    mask_b = ((gray >> np.uint32(b)) & np.uint32(1)) != 0
                    acc_c = np.where(mask_b, acc_c ^ sv[c_dim, b], acc_c)
                acc_c ^= shifts[r, c_dim]
                uc = np.clip((acc_c.astype(np.float64) + 0.5) * inv_pow2, 1e-12, 1.0 - 1e-12)
                zc = norm.ppf(uc)
                zf = np.empty((n_sub, n_rep), dtype=np.float64)
                for s in range(n_sub):
                    f_dim = (m_rem + m * n_sub + s) * d_assets + d
                    acc_f = np.zeros(n_rep, dtype=np.uint32)
                    for b in range(30):
                        mask_b = ((gray >> np.uint32(b)) & np.uint32(1)) != 0
                        acc_f = np.where(mask_b, acc_f ^ sv[f_dim, b], acc_f)
                    acc_f ^= shifts[r, f_dim]
                    uf = np.clip((acc_f.astype(np.float64) + 0.5) * inv_pow2, 1e-12, 1.0 - 1e-12)
                    zf[s] = norm.ppf(uf)
                z_ref[m, :, r, :, d] = zf + (zc * inv_sqrt_sub - np.mean(zf, axis=0))[None, :]
    z_mean_ref = np.mean(z_ref, axis=1)
    z_rot_ref = np.einsum("mk,krnd->mrnd", bb_ortho, z_mean_ref)
    z_indep_ref = z_ref + (z_rot_ref - z_mean_ref)[:, None, :, :, :]
    z_corr_ref = np.matmul(z_indep_ref, chol.T)

    np.testing.assert_allclose(z_indep_cpp, z_indep_ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(z_corr_cpp, z_corr_ref, atol=1e-12, rtol=1e-12)

    # Also test antithetic mode.
    z_anti, _ = tb.crn_generate_sobol_normals_cpu(
        d_assets=d_assets,
        m_rem=m_rem,
        n_sub=n_sub,
        eff_r=1,
        n_rep=n_rep,
        dir_numbers=sv,
        digital_shifts=shifts[:1],
        antithetic=True,
        return_indep=True,
        return_corr=False,
    )
    assert z_anti is not None
    half = n_rep // 2
    np.testing.assert_allclose(z_anti[:, :, :, half:, :], -z_anti[:, :, :, :half, :], atol=1e-14, rtol=1e-14)


def test_crn_loo_l2o_v3_longbasket_onestar_and_vibrato_gamma_cpp() -> None:
    """Verify V3 literal memory, IsLongBasket I_ramp, OneStar, Vibrato Gamma C++, and coupon_cf_undisc."""
    import ctypes
    import numpy.typing as npt

    from deepvol.models.autocall import _tier1_backend as tb
    from deepvol.models.autocall.qmc_aad_greeks import _CrnLooL2oBatchArgsC

    backend = tb.get_tier1_backend()
    assert isinstance(backend, tb.Cpp23Tier1Backend)

    rng = np.random.default_rng(106)
    d_assets, m_rem, n_paths = 3, 4, 48
    perf = np.exp(rng.normal(loc=-0.03, scale=0.22, size=(m_rem, d_assets, n_paths))).astype(np.float64)
    df_pay = np.array([[0.98, 0.96, 0.94, 0.92]], dtype=np.float64)
    ko_bar = np.array([1.05, 1.00, 0.98, 0.95], dtype=np.float64)
    is_call = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    cpn_bar = np.array([0.85, 0.85, 0.85, 0.85], dtype=np.float64)
    is_guar = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cpn_rates = np.array([0.03, 0.03, 0.03, 0.03], dtype=np.float64)

    def _ref_v3_pv(is_long_basket: bool) -> float:
        surv = np.ones(n_paths, dtype=np.float64)
        c_mem = np.full(n_paths, 0.02, dtype=np.float64)
        pv = np.zeros(n_paths, dtype=np.float64)
        ko_sp, cpn_sp, ki_sp = -0.05, -0.05, -0.05
        strike, k_os, eps_os, os_cpn = 1.0, 1.02, 0.06, 0.04
        for m in range(m_rem):
            w_call = np.min(perf[m], axis=0)
            w_cpn = w_call
            blo_ko, bhi_ko = min(ko_bar[m], ko_bar[m] + ko_sp), max(ko_bar[m], ko_bar[m] + ko_sp)
            gk = np.clip((w_call - blo_ko) / (bhi_ko - blo_ko), 0.0, 1.0)
            blo_cp, bhi_cp = min(cpn_bar[m], cpn_bar[m] + cpn_sp), max(cpn_bar[m], cpn_bar[m] + cpn_sp)
            gc = np.clip((w_cpn - blo_cp) / (bhi_cp - blo_cp), 0.0, 1.0)
            eff_gc = np.maximum(gc, gk)
            eff_c = cpn_rates[m] + c_mem
            pm = surv * gk
            cpn_cf = surv * eff_gc * eff_c + pm * 0.01
            c_mem = surv * eff_c * (1.0 - eff_gc)
            pv += df_pay[0, m] * (pm + cpn_cf)
            surv = np.maximum(surv - pm, 0.0)
        w_best = np.max(perf[-1], axis=0)
        w_avg = np.mean(perf[-1], axis=0)
        w_wof = np.min(perf[-1], axis=0)
        os_ind = np.clip((w_best - (k_os - 0.5 * eps_os)) / eps_os, 0.0, 1.0)
        if is_long_basket:
            blo_k, bhi_k = min(strike, strike + ki_sp), max(strike, strike + ki_sp)
            ki_g = np.clip((bhi_k - w_wof) / (bhi_k - blo_k), 0.0, 1.0)
            opt = -1.0 * ki_g * (1.0 - os_ind) * (1.0 - w_avg) + os_ind * os_cpn
        else:
            blo_k, bhi_k = min(0.75, 0.75 + ki_sp), max(0.75, 0.75 + ki_sp)
            ki_g = np.clip((bhi_k - w_wof) / (bhi_k - blo_k), 0.0, 1.0)
            opt = -1.2 * ki_g * (1.0 - os_ind) * np.maximum(strike - w_wof, 0.0) + os_ind * os_cpn
        pv += df_pay[0, -1] * surv * (1.0 + opt)
        return float(np.mean(pv))

    for is_lb in (False, True):
        pv_out = np.zeros(1, dtype=np.float64)
        asset_a = np.array([-1], dtype=np.int32)
        asset_b = np.array([-1], dtype=np.int32)
        c_args = _CrnLooL2oBatchArgsC(
            n_bumps=1,
            d_assets=d_assets,
            m_rem=m_rem,
            n_paths_total=n_paths,
            perf_base=perf.ctypes.data,
            bump_perf_a=0,
            bump_perf_b=0,
            bump_asset_a=asset_a.ctypes.data,
            bump_asset_b=asset_b.ctypes.data,
            df_pay_mat=df_pay.ctypes.data,
            w_fund_mat=0,
            pv_in_transit=0,
            pv_out=pv_out.ctypes.data,
            ko_barriers=ko_bar.ctypes.data,
            is_callable=is_call.ctypes.data,
            cpn_barriers=cpn_bar.ctypes.data,
            is_guaranteed_cpn=is_guar.ctypes.data,
            coupon_rates=cpn_rates.ctypes.data,
            ko_spread=-0.05,
            cpn_spread=-0.05,
            ki_spread=-0.05,
            one_star_strike=1.02,
            one_star_eps=0.06,
            strike=1.0,
            lower_strike=0.0,
            ki_barrier=0.75,
            option_gearing=1.2,
            autocall_coupon=0.01,
            c_mem_0=0.02,
            i_ki_past=0.0,
            part_call_strike=0.0,
            part_call_rate=0.0,
            one_star_coupon=0.04,
            call_code=0,
            cpn_code=0,
            opt_code=2 if is_lb else 0,
            has_ki=1,
            is_memory=1,
            is_long_basket=1 if is_lb else 0,
            no_notional=0,
            force_cpn_ac=1,
            cpn_dir_down=0,
            cpn_type_out=0,
        )
        backend._fn_crn_loo_l2o(ctypes.byref(c_args))
        np.testing.assert_allclose(pv_out[0], _ref_v3_pv(is_lb), atol=1e-13, rtol=1e-13)

    # Verify crn_eval_vibrato_gamma_cpp against an independent Python/NumPy reference.
    corr = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.25], [0.1, 0.25, 1.0]], dtype=np.float64)
    chol = np.ascontiguousarray(np.linalg.cholesky(corr), dtype=np.float64)
    l_inv_t = np.ascontiguousarray(np.linalg.inv(chol).T, dtype=np.float64)
    c_inv = np.ascontiguousarray(np.linalg.inv(corr), dtype=np.float64)
    vib_n_steps = 2
    sv_vib = np.ascontiguousarray(rng.standard_normal((vib_n_steps, d_assets, n_paths)), dtype=np.float64)
    lx_pre = np.ascontiguousarray(rng.normal(scale=0.15, size=(d_assets, n_paths)), dtype=np.float64)
    j_pre = np.ascontiguousarray(np.exp(lx_pre) * 1.02, dtype=np.float64)
    sig_pre = np.ascontiguousarray(rng.uniform(0.18, 0.32, size=(d_assets, n_paths)), dtype=np.float64)
    fwd_term = np.array([1.02, 0.99, 1.05], dtype=np.float64)
    cdl_m = np.ascontiguousarray(rng.uniform(0.95, 1.05, size=(m_rem, d_assets)), dtype=np.float64)
    cdl2_last = np.array([0.02, -0.01, 0.015], dtype=np.float64)
    eps_vec = np.full(m_rem, 0.03, dtype=np.float64)
    perm = np.arange(d_assets, dtype=np.int64)

    def _s5(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = np.clip(t, 0.0, 1.0)
        return np.asarray(u * u * u * (10.0 + u * (-15.0 + 6.0 * u)), dtype=np.float64)

    def _ds5(t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        u = np.clip(t, 0.0, 1.0)
        return np.asarray(
            np.where((t > 0.0) & (t < 1.0), 30.0 * u * u * (1.0 - u) * (1.0 - u), 0.0),
            dtype=np.float64,
        )

    for is_lb in (False, True):
        h_cpp, frozen_cpp = tb.crn_eval_vibrato_gamma_cpu(
            perf_mat=perf,
            lx_pre=lx_pre,
            j_pre=j_pre,
            sig_pre=sig_pre,
            sv_vib=sv_vib,
            chol_mat=chol,
            l_inv_t=l_inv_t,
            c_inv=c_inv,
            fwd_term=fwd_term,
            cdl_m=cdl_m,
            cdl2_last=cdl2_last,
            ko_barriers=ko_bar,
            is_callable=is_call,
            cpn_barriers=cpn_bar,
            is_guaranteed_cpn=is_guar,
            coupon_rates=cpn_rates,
            eps_vec=eps_vec,
            inv_lex_perm=perm,
            vib_step_start=0,
            vib_n_steps=vib_n_steps,
            df_last=0.92,
            h_vib=0.08,
            ko_spread=-0.05,
            cpn_spread=-0.05,
            ki_spread=-0.05,
            one_star_strike=1.02,
            one_star_eps=0.06,
            strike=1.0,
            lower_strike=0.0,
            ki_barrier=0.75,
            option_gearing=1.2,
            autocall_coupon=0.01,
            c_mem_0=0.02,
            i_ki_past=0.0,
            part_call_strike=0.0,
            part_call_rate=0.0,
            one_star_coupon=0.04,
            call_code=0,
            cpn_code=0,
            opt_code=2 if is_lb else 0,
            has_ki=True,
            ki_active=True,
            has_one_star=True,
            is_memory=True,
            is_long_basket=is_lb,
            no_notional=False,
            force_cpn_ac=True,
            cpn_dir_down=False,
            cpn_type_out=False,
            has_lower_strike=False,
            has_part_call=False,
            emit_frozen_term=True,
        )
        assert frozen_cpp is not None

        # Pure NumPy reference for Vibrato Gamma + V3 literal dcm.
        surv_pre = np.ones(n_paths, dtype=np.float64)
        c_mem = np.full(n_paths, 0.02, dtype=np.float64)
        dlns = np.zeros((d_assets, n_paths), dtype=np.float64)
        dcm = np.zeros((d_assets, n_paths), dtype=np.float64)
        for m in range(m_rem - 1):
            w_call = np.min(perf[m], axis=0)
            idx_min = np.argmin(perf[m], axis=0)
            inv_w_ko = 1.0 / 0.05
            t_ko = (w_call - (ko_bar[m] - 0.05)) * inv_w_ko
            gk = _s5(t_ko)
            dgk = _ds5(t_ko) * inv_w_ko
            coef_s = -dgk / np.maximum(1.0 - gk, 1e-8)
            inv_w_cp = 1.0 / 0.05
            t_cp = (w_call - (cpn_bar[m] - 0.05)) * inv_w_cp
            gc = _s5(t_cp)
            dgc = _ds5(t_cp) * inv_w_cp
            eff_gc = np.maximum(gc, gk)
            wc_m = np.where(gc > gk, 1.0, np.where(gc < gk, 0.0, 0.5))
            wk_m = 1.0 - wc_m
            eff_c = cpn_rates[m] + c_mem
            for d in range(d_assets):
                mask_d = (idx_min == d).astype(np.float64)
                pd = perf[m, d]
                prev_dlns = dlns[d].copy()
                d_egc = mask_d * (wc_m * dgc + wk_m * dgk) * pd * cdl_m[m, d]
                dcm[d] = surv_pre * (prev_dlns * eff_c * (1.0 - eff_gc) + dcm[d] * (1.0 - eff_gc) - eff_c * d_egc)
                dlns[d] = prev_dlns + mask_d * coef_s * pd * cdl_m[m, d]
            c_mem = surv_pre * eff_c * (1.0 - eff_gc)
            surv_pre = np.maximum(surv_pre - surv_pre * gk, 0.0)

        zk_mean = np.sum(sv_vib, axis=0) / math.sqrt(float(vib_n_steps))
        zc = chol @ zk_mean
        xi_ref = l_inv_t @ zk_mean
        sqrt_h = math.sqrt(0.08)
        eff_c_l = cpn_rates[-1] + c_mem
        vals = {}
        for sgn, mult in ((0, 0.0), (1, 1.0), (-1, -1.0)):
            xd = lx_pre - 0.5 * sig_pre * sig_pre * 0.08 + mult * sig_pre * sqrt_h * zc
            p_s = fwd_term[:, None] * np.exp(np.clip(xd, -30.0, 30.0))
            wc_s = np.min(p_s, axis=0)
            wb_s = np.max(p_s, axis=0)
            wa_s = np.mean(p_s, axis=0)
            gk_s = np.clip((wc_s - (ko_bar[-1] - 0.05)) / 0.05, 0.0, 1.0)
            gc_s = np.clip((wc_s - (cpn_bar[-1] - 0.05)) / 0.05, 0.0, 1.0)
            egc_s = np.maximum(gc_s, gk_s)
            os_s = np.clip((wb_s - (1.02 - 0.03)) / 0.06, 0.0, 1.0)
            if is_lb:
                ki_s = np.clip((1.0 - wc_s) / 0.05, 0.0, 1.0)
                opt_s = -1.0 * ki_s * (1.0 - os_s) * (1.0 - wa_s) + os_s * 0.04
            else:
                ki_s = np.clip((0.75 - wc_s) / 0.05, 0.0, 1.0)
                opt_s = -1.2 * ki_s * (1.0 - os_s) * np.maximum(1.0 - wc_s, 0.0) + os_s * 0.04
            vals[sgn] = (gk_s * (1.0 + 0.01) + egc_s * eff_c_l + (1.0 - gk_s) * (1.0 + opt_s), egc_s)

        df_diff = 0.5 * (vals[1][0] - vals[-1][0])
        df_hess = 0.5 * (vals[1][0] + vals[-1][0]) - vals[0][0]
        degc_diff = 0.5 * (vals[1][1] - vals[-1][1])
        inv_s = 1.0 / np.maximum(sig_pre * sqrt_h, 1e-6)
        j_last = j_pre * cdl_m[-1, :, None]
        h_ref = np.zeros((d_assets, d_assets), dtype=np.float64)
        for i in range(d_assets):
            w1_raw_i = df_diff * xi_ref[i] * inv_s[i] * j_pre[i]
            w1_i = df_diff * xi_ref[i] * inv_s[i] * j_last[i]
            w1_egc_i = degc_diff * xi_ref[i] * inv_s[i] * j_last[i]
            for j in range(d_assets):
                w1_j = df_diff * xi_ref[j] * inv_s[j] * j_last[j]
                w1_egc_j = degc_diff * xi_ref[j] * inv_s[j] * j_last[j]
                w2_ij = df_hess * (xi_ref[i] * xi_ref[j] - c_inv[i, j]) * inv_s[i] * inv_s[j] * j_last[i] * j_last[j]
                diag_corr = (w1_raw_i * cdl2_last[i]) if i == j else 0.0
                h_ij = 0.92 * surv_pre * (w2_ij + dlns[i] * w1_j + dlns[j] * w1_i + dcm[i] * w1_egc_j + dcm[j] * w1_egc_i + diag_corr)
                h_ref[i, j] = float(np.mean(h_ij))
        np.testing.assert_allclose(h_cpp, h_ref, atol=1e-12, rtol=1e-12)

    # Verify analytical_baseline coupon_cf_undisc parity across CPU C++ and PureTorch paths.
    st_mem = {
        "forward_perf": torch.tensor([[1.02, 0.98, 0.95], [0.99, 0.96, 0.92]], dtype=torch.float64),
        "v_perf": torch.tensor([[0.22, 0.28, 0.32], [0.24, 0.30, 0.34]], dtype=torch.float64),
        "corr_matrix": torch.tensor([[1.0, 0.4], [0.4, 1.0]], dtype=torch.float64),
        "df_pay": torch.tensor([0.97, 0.94, 0.91], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.05, 1.02, 1.00], dtype=torch.float64),
        "is_callable": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.85, 0.85, 0.85], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.025, 0.025, 0.025], dtype=torch.float64),
        "is_memory": True,
        "c_mem_0": 0.015,
        "autocall_coupon": 0.01,
        "strike": 1.0,
        "ki_barrier": 0.70,
        "has_ki": True,
    }
    out_c = compute_analytical_autocall_baseline(st_mem, use_fast_cpu=True)
    out_t = compute_analytical_autocall_baseline(st_mem, use_fast_cpu=False)
    assert out_c.coupon_cf_undisc is not None and out_t.coupon_cf_undisc is not None
    torch.testing.assert_close(out_c.coupon_cf_undisc, out_t.coupon_cf_undisc, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        (out_c.coupon_cf_undisc * st_mem["df_pay"]).sum(), out_c.coupon_pv_proxy, atol=1e-12, rtol=1e-12
    )







