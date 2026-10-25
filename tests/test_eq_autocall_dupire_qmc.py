"""test_eq_autocall_dupire_qmc.py — Verification suite for Task 1.3 (Durrleman Regularizer & Dupire QMC Solver).

Covers:
  - Gate G1.2: Multi-asset Dupire Local Volatility QMC reference trade reproduction on
    `eq_autocallable_request.txt` (`Option = -0.7412979343448728`, `ProductValue = 0.2181930543553945`
    within <= 1.0 * StandardError = 0.00029138).
  - Duplicate strike deduplication (`VKCO_RUB` flat tail pillars) and calendar total-variance
    monotonicity projection (d_T w >= 1e-5).
  - Exact analytical quote-grid Jacobians `.iv_and_d_iv_d_quotes` and `.extract_node_surface_features`
    verified against central finite differences.
  - Dupire local volatility clamping in [0.01, 2.50] and `SMILE_ARBITRAGE` rejection on corrupt surfaces.
  - Agreement between NumPy and PyTorch evaluators across all 14 Murex basket monitoring indices.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deepvol.api.schemas_eq_autocall import EQAutocallableTradeRequest, parse_murex_autocall_dag
from deepvol.models.autocall import (
    RegularizedVolSurface,
    enforce_no_arbitrage_projection,
    evaluate_monitoring_index_np,
    evaluate_monitoring_index_torch,
    preprocess_eq_autocall,
    price_eq_autocall_dupire_qmc,
    price_eq_autocall_qmc_fallback,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = REPO_ROOT / "eq_autocallable_request.txt"


@pytest.fixture(scope="module")
def parsed_dag() -> tuple[EQAutocallableTradeRequest, dict[str, Any]]:
    return parse_murex_autocall_dag(REF_REQUEST_PATH)


def test_G1_2_dupire_qmc_reference_trade_reproduction(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G1.2: Reproduce Murex reference Option & ProductValue within <= 1.0 * StandardError."""
    req_model, expected_res = parsed_dag
    ref_pv = float(expected_res["ProductValue"])
    ref_opt = float(expected_res["Option"])
    ref_se = float(expected_res["StandardError"])

    assert ref_pv == pytest.approx(0.2181930543553945, abs=1e-15)
    assert ref_opt == pytest.approx(-0.7412979343448728, abs=1e-15)
    assert ref_se == pytest.approx(0.0002913772720117921, abs=1e-15)

    state = preprocess_eq_autocall(req_model)

    # 1. Dupire Local Volatility QMC mode (with Brownian bridge sub-stepping)
    res_lv = price_eq_autocall_dupire_qmc(state, n_paths=32768, mode="dupire_lv")
    err_opt_lv = abs(res_lv.option_pv - ref_opt)
    err_pv_lv = abs(res_lv.product_value - ref_pv)

    assert err_opt_lv <= 1.0 * ref_se, (
        f"Gate G1.2 failed (dupire_lv Option): |{res_lv.option_pv} - {ref_opt}| = {err_opt_lv:.6f} > {ref_se:.6f}"
    )
    assert err_pv_lv <= 1.0 * ref_se, (
        f"Gate G1.2 failed (dupire_lv ProductValue): |{res_lv.product_value} - {ref_pv}| = {err_pv_lv:.6f} > {ref_se:.6f}"
    )
    assert len(res_lv.cashflows) == 37
    assert res_lv.notional_pv == pytest.approx(0.9594909887002673, abs=1e-12)
    assert res_lv.coupon_pv == pytest.approx(0.0, abs=1e-12)

    # 2. Terminal IV QMC mode
    res_iv = price_eq_autocall_dupire_qmc(state, n_paths=32768, mode="terminal_iv")
    assert abs(res_iv.option_pv - ref_opt) <= 1.0 * ref_se
    assert abs(res_iv.product_value - ref_pv) <= 1.0 * ref_se

    # 3. Deterministic fallback entrypoint alias
    res_fb = price_eq_autocall_qmc_fallback(req_model, n_paths=32768)
    assert abs(res_fb.product_value - ref_pv) <= 1.0 * ref_se


def test_regularized_vol_surface_deduplication_and_calendar_monotonicity(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify VKCO_RUB flat tail handling, strike deduplication, and calendar variance monotonicity."""
    req_model, _ = parsed_dag
    state = preprocess_eq_autocall(req_model)

    vkco_surf = state.vol_surfaces["VKCO_RUB"]
    assert np.all(np.diff(vkco_surf.strikes) > 0.0)
    # VKCO_RUB has identical quote values on the last two strikes (845 and 870) and last two tenors
    np.testing.assert_allclose(vkco_surf.quotes[:, -2], vkco_surf.quotes[:, -1], rtol=1e-12, atol=1e-12)

    # Total variance w(T_m, K) must be strictly increasing across all tenors
    w_grid = (vkco_surf.quotes**2) * vkco_surf.tenors[:, None]
    assert np.all(np.diff(w_grid, axis=0) > 0.0)

    # Synthetic surface with duplicate strikes [80, 100, 120, 120] and calendar variance violation
    tenors = np.array([0.25, 0.50, 1.00], dtype=np.float64)
    strikes = np.array([80.0, 100.0, 120.0, 120.0], dtype=np.float64)
    quotes = np.array(
        [
            [0.40, 0.35, 0.32, 0.34],
            [0.20, 0.18, 0.17, 0.17],  # Calendar violation: w(0.5) < w(0.25)
            [0.35, 0.30, 0.28, 0.28],
        ],
        dtype=np.float64,
    )
    u_k, proj_q, dedup_m, was_proj = enforce_no_arbitrage_projection(tenors, strikes, quotes)
    assert was_proj is True
    assert len(u_k) == 3
    assert dedup_m.shape == (3, 4)
    w_proj = (proj_q**2) * tenors[:, None]
    assert np.all(np.diff(w_proj, axis=0) >= 1e-5 * 0.25 - 1e-12)


def test_regularized_vol_surface_analytic_jacobians_vs_fd(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify analytical quote Jacobians (.iv_and_d_iv_d_quotes & .extract_node_surface_features) vs FD."""
    req_model, _ = parsed_dag
    as_of_dt = req_model.Model.MarketDataSet.AsOfDate
    magn_spec = req_model.Model.MarketDataSet.VolatilitySurfaces["MAGN_RUB"]
    surf = RegularizedVolSurface(magn_spec, as_of_date=as_of_dt)

    t_query = 0.25
    k_query = 28.5
    iv_val, jac_iv = surf.iv_and_d_iv_d_quotes(t_query, k_query)
    assert jac_iv.shape == surf.raw_quotes.shape
    assert iv_val > 0.01

    eps = 1e-6
    jac_fd = np.zeros_like(jac_iv)
    # Test finite differences across the active maturity bracket rows
    for r in range(min(3, surf.n_exp)):
        for c in range(surf.n_raw_strikes):
            spec_up = copy.deepcopy(magn_spec)
            spec_dn = copy.deepcopy(magn_spec)
            spec_up.VolatilityQuotes[r][c] += eps
            spec_dn.VolatilityQuotes[r][c] -= eps
            iv_u = float(RegularizedVolSurface(spec_up, as_of_date=as_of_dt).iv(t_query, k_query))
            iv_d = float(RegularizedVolSurface(spec_dn, as_of_date=as_of_dt).iv(t_query, k_query))
            jac_fd[r, c] = (iv_u - iv_d) / (2.0 * eps)

    np.testing.assert_allclose(jac_iv[:3, :], jac_fd[:3, :], rtol=1e-5, atol=1e-7)

    # Also verify 7D node surface feature Jacobian
    feats_7d, jac_7d = surf.extract_node_surface_features(0.15, 27.5, 0.26, 28.8)
    assert feats_7d.shape == (7,)
    assert jac_7d.shape == (7, surf.n_exp, surf.n_raw_strikes)

    for r in range(min(2, surf.n_exp)):
        for c in range(min(4, surf.n_raw_strikes)):
            spec_up = copy.deepcopy(magn_spec)
            spec_dn = copy.deepcopy(magn_spec)
            spec_up.VolatilityQuotes[r][c] += eps
            spec_dn.VolatilityQuotes[r][c] -= eps
            f_u, _ = RegularizedVolSurface(spec_up, as_of_date=as_of_dt).extract_node_surface_features(
                0.15, 27.5, 0.26, 28.8
            )
            f_d, _ = RegularizedVolSurface(spec_dn, as_of_date=as_of_dt).extract_node_surface_features(
                0.15, 27.5, 0.26, 28.8
            )
            fd_col = (f_u - f_d) / (2.0 * eps)
            np.testing.assert_allclose(jac_7d[:, r, c], fd_col, rtol=1e-4, atol=1e-6)


def test_regularized_vol_surface_local_vol_bounds_and_smile_arbitrage_rejection(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify local_vol stays clamped in [0.01, 2.50] and SMILE_ARBITRAGE is raised on corrupt inputs."""
    req_model, _ = parsed_dag
    state = preprocess_eq_autocall(req_model)

    for ticker, surf in state.vol_surfaces.items():
        s_grid = np.linspace(0.1 * surf.strikes[0], 2.5 * surf.strikes[-1], 50, dtype=np.float64)
        lv = surf.local_vol(0.20, s_grid, forward_at_t=float(np.mean(surf.strikes)), cash_div_leverage=1.05)
        assert lv.dtype == np.float64
        assert np.all(lv >= 0.01) and np.all(lv <= 2.50), f"Local vol out of bounds for {ticker}"

    # Corrupt surface checks -> must raise ValueError with SMILE_ARBITRAGE prefix
    with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
        enforce_no_arbitrage_projection(
            np.array([0.5, 1.0]),
            np.array([100.0, 110.0]),
            np.array([[-0.20, 0.25], [0.25, 0.25]]),
        )

    with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
        enforce_no_arbitrage_projection(
            np.array([0.5, 1.0]),
            np.array([100.0, 110.0]),
            np.array([[0.25, np.nan], [0.25, 0.25]]),
        )

    with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
        enforce_no_arbitrage_projection(
            np.array([1.0, 0.5]),  # Non-increasing tenors
            np.array([100.0, 110.0]),
            np.array([[0.25, 0.25], [0.25, 0.25]]),
        )


@pytest.mark.parametrize(
    "index_name",
    [
        "WorstOf",
        "BestOf",
        "Average",
        "Dispersion",
        "RatioBestWorst",
        "RatioBestAverage",
        "RatioAverageWorst",
        "DiffBestWorst",
        "DiffBestAverage",
        "DiffAverageWorst",
        "RatioAB",
        "RatioBA",
        "DiffAB",
        "DiffBA",
        "Outperformance",
        "TwinWin",
        "Rainbow",
        "Lookback",
        "Spread",
    ],
)
def test_all_monitoring_indices_np_vs_torch(index_name: str) -> None:
    """Verify exact float64 agreement between NumPy and PyTorch across all basket monitoring indices."""
    perf_np = np.array(
        [
            [0.82, 0.95, 1.08, 0.74],
            [1.12, 1.03, 0.91, 1.25],
        ],
        dtype=np.float64,
    )
    perf_t = torch.from_numpy(perf_np)

    out_np = np.asarray(evaluate_monitoring_index_np(perf_np, index_name), dtype=np.float64)
    out_t = evaluate_monitoring_index_torch(perf_t, index_name).numpy()
    np.testing.assert_allclose(out_np, out_t, rtol=1e-13, atol=1e-13)
