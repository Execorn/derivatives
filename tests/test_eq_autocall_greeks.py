"""test_eq_autocall_greeks.py — Verification suite for Task 3.1 (Differentiable Chain-Rule Greeks Engine).

Covers:
  - Gate G1.8: Analytical/Autograd `Delta`, `Rho`, `DividendsSensitivities` (plus `Vega`, `Cega`, `Gamma`)
    vs Central Finite-Difference Oracle (Relative Error <= 1e-5; achieved <= 1e-7).
  - Gate G1.9: Directional Euler CashGamma Identity (`MultivariateAggregatedCashGamma` via O(1) 2-JVP
    `compute_euler_cash_gamma` vs full Hessian quadratic form sum_{i,j} S_i S_j Gamma_{ij}, |Delta| <= 1e-9).
  - Proof W2: Cash-dividend leverage and sticky-strike forward-vol contributions to Spot Delta and CashGamma.
  - S_D Unpermutation (pi_lex^{-1}): Exact restoration of caller asset order across all per-asset Greeks.
  - Gate G2.4 & REM-S0-01: Vectorized batched GPU/CPU Greeks (`compute_batched_surrogate_greeks`) and
    strict float64 outer / float32 inner precision boundary segregation.
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pytest
import torch

from deepvol.api.schemas_eq_autocall import (
    EQAutocallableTradeRequest,
    MonitoringIndexEnum,
    PredefinedDividendsSpec,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall.greeks_engine import (
    AutocallGreeksResult,
    compute_batched_surrogate_greeks,
    compute_eq_autocall_greeks,
    evaluate_differentiable_pv,
)
from deepvol.models.autocall.pre_post_processor import preprocess_eq_autocall
from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


@pytest.fixture(scope="module")
def parsed_dag() -> tuple[EQAutocallableTradeRequest, dict[str, Any]]:
    return parse_murex_autocall_dag(REF_REQUEST_PATH)


@pytest.fixture(scope="module")
def multi_period_request(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> EQAutocallableTradeRequest:
    """Constructs a 5-asset, 3-future-period seasoned request with active KO barriers and future cash dividends."""
    req_orig, _ = parsed_dag
    req = copy.deepcopy(req_orig)
    # Shift AsOfDate back to 2026-06-01 so Periods 10, 11, 12 are all future active periods (M_rem = 3)
    # and bring spots closer to reference spots so KO, coupon, put, and dividend sensitivities are all large.
    req.Model.MarketDataSet.AsOfDate = datetime.datetime(
        2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc
    )
    for curve in req.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)

    # Set distinct spots across 74-93% of reference spots so worst-of ordering is non-degenerate
    # and early autocall, memory coupons, and terminal put are all active
    moneynesses = [0.74, 0.79, 0.83, 0.88, 0.93]
    for idx, (uid, ref_s) in enumerate(
        zip(req.Product.UnderlyingIdentifiers, req.Product.UnderlyingReferenceSpots)
    ):
        req.Model.MarketDataSet.Spots[uid] = moneynesses[idx % len(moneynesses)] * float(ref_s)

    # Ensure Period 10 & 11 have callable KO barriers (0.95, 0.92) and conditional coupon barriers (0.78)
    req.Product.AutoCallTriggerLevels[9] = 0.95
    req.Product.AutoCallTriggerLevels[10] = 0.92
    req.Product.CouponBarrierLevels[9] = 0.78
    req.Product.CouponBarrierLevels[10] = 0.78

    # Add explicit future discrete dividends on the two worst-of assets (VKCO_RUB and YDEX_RUB) before maturity
    req.Model.MarketDataSet.Dividends = {}
    req.Product.PredefinedDividends = {
        "VKCO_RUB": PredefinedDividendsSpec(
            ExDates=[
                datetime.datetime(2026, 8, 15, 0, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2026, 12, 10, 0, 0, tzinfo=datetime.timezone.utc),
            ],
            Amounts=[18.0, 24.0],
        ),
        "YDEX_RUB": PredefinedDividendsSpec(
            ExDates=[
                datetime.datetime(2026, 9, 20, 0, 0, tzinfo=datetime.timezone.utc),
            ],
            Amounts=[65.0],
        ),
    }
    return req


def _rel_err(got: np.ndarray | torch.Tensor, ref: np.ndarray | torch.Tensor) -> float:
    g = np.asarray(got, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    denom = max(float(np.linalg.norm(r)), 1e-10)
    return float(np.linalg.norm(g - r) / denom)


def test_G1_8_autograd_delta_rho_dividends_vs_central_finite_difference(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Gate G1.8: Analytical/Autograd Delta, Rho, DividendsSensitivities (plus Vega & Cega) vs Central FD."""
    req_ref, _ = parsed_dag

    torch.manual_seed(42)
    surrogate_f64 = MaskedEQAutocallSurrogate(
        d_max=8, m_max=48, d_model=128, n_egno_layers=3, n_sched_layers=2, n_heads=4
    ).to(dtype=torch.float64).eval()

    torch.manual_seed(42)
    surrogate_f32 = MaskedEQAutocallSurrogate(
        d_max=8, m_max=48, d_model=128, n_egno_layers=3, n_sched_layers=2, n_heads=4
    ).eval()
    surrogate_f32.zero_init_residual_heads(std=1e-3)
    assert not surrogate_f32.is_zero_residual()
    surrogate_f32_ref64 = copy.deepcopy(surrogate_f32).to(dtype=torch.float64).eval()

    for req in (req_ref, multi_period_request):
        state = preprocess_eq_autocall(req)
        d_assets = state.d_assets
        sorted_ids = state.sorted_identifiers
        pv_base_none: float | None = None

        for surr in (None, surrogate_f64, surrogate_f32):
            res = compute_eq_autocall_greeks(
                state,
                queries=[
                    "ProductValue",
                    "Delta",
                    "Rho",
                    "DividendsSensitivities",
                    "Vega",
                    "Cega",
                ],
                surrogate=surr,
            )
            if surr is None:
                pv_base_none = float(res.product_value)
            elif surr is surrogate_f32:
                assert pv_base_none is not None
                assert abs(float(res.product_value) - pv_base_none) > 1e-6
                res_ref64 = compute_eq_autocall_greeks(
                    state,
                    queries=["ProductValue", "Delta", "Rho", "Vega", "Cega"],
                    surrogate=surrogate_f32_ref64,
                )
                assert res_ref64.delta_tensor is not None and res.delta_tensor is not None
                assert _rel_err(res.delta_tensor.detach().cpu(), res_ref64.delta_tensor.detach().cpu()) <= 1e-6
                assert res_ref64.rho_arrays is not None and res.rho_arrays is not None
                for cid_ref, rho_ref in res_ref64.rho_arrays.items():
                    assert _rel_err(res.rho_arrays[cid_ref], rho_ref) <= 1e-6
                assert res_ref64.vega_arrays is not None and res.vega_arrays is not None
                for tk_ref, v_ref in res_ref64.vega_arrays.items():
                    assert _rel_err(res.vega_arrays[tk_ref], v_ref) <= 2.5e-5
                assert res_ref64.cega_tensor is not None and res.cega_tensor is not None
                assert _rel_err(res.cega_tensor.detach().cpu(), res_ref64.cega_tensor.detach().cpu()) <= 1e-6

            assert res.delta_tensor is not None
            assert res.rho_arrays is not None
            assert res.dividends_arrays is not None
            assert res.vega_arrays is not None
            assert res.cega_tensor is not None

            # FP32 forward pass requires larger central FD steps to avoid subtraction truncation noise.
            eps_s = 1e-3 if surr is surrogate_f32 else 1e-5
            fd_delta_sorted = np.zeros(d_assets, dtype=np.float64)
            base_spots = state.spots_t0.detach().clone()
            for i in range(d_assets):
                h = float(base_spots[i].item()) * eps_s
                sp_up = base_spots.clone()
                sp_dn = base_spots.clone()
                sp_up[i] += h
                sp_dn[i] -= h
                pv_up = float(evaluate_differentiable_pv(state, spots_t0=sp_up, surrogate=surr).item())
                pv_dn = float(evaluate_differentiable_pv(state, spots_t0=sp_dn, surrogate=surr).item())
                fd_delta_sorted[i] = (pv_up - pv_dn) / (2.0 * h)

            fd_delta_orig = fd_delta_sorted[state.inv_lex_perm]
            got_delta_orig = res.delta_tensor.detach().cpu().numpy()
            rel_delta = _rel_err(got_delta_orig, fd_delta_orig)
            assert rel_delta <= 1e-5, f"Delta relative error {rel_delta:.3e} > 1e-5"
            if surr is not surrogate_f32:
                assert rel_delta <= 1e-7, f"Float64 Delta relative error {rel_delta:.3e} > 1e-7"

            # 2. Rho vs Central Finite Differences on discount curve pillar quotes
            curve_id = next(iter(res.rho_arrays.keys()))
            got_rho = res.rho_arrays[curve_id]
            k_pillars = len(got_rho)
            fd_rho = np.zeros(k_pillars, dtype=np.float64)
            h_rate = 5e-2 if surr is surrogate_f32 else 1e-4
            for k in range(k_pillars):
                p_up = torch.zeros(k_pillars, dtype=torch.float64)
                p_dn = torch.zeros(k_pillars, dtype=torch.float64)
                p_up[k] = h_rate
                p_dn[k] = -h_rate
                pv_up = float(
                    evaluate_differentiable_pv(state, curve_pillar_deltas=p_up, surrogate=surr).item()
                )
                pv_dn = float(
                    evaluate_differentiable_pv(state, curve_pillar_deltas=p_dn, surrogate=surr).item()
                )
                fd_rho[k] = (pv_up - pv_dn) / (2.0 * h_rate)

            rel_rho = _rel_err(got_rho, fd_rho)
            assert rel_rho <= 1e-5, f"Rho relative error {rel_rho:.3e} > 1e-5"
            if surr is not surrogate_f32:
                assert rel_rho <= 1e-7, f"Float64 Rho relative error {rel_rho:.3e} > 1e-7"

            # Also verify Rho against full request re-preprocessing when bumping RatesSchedule.Rates[k]
            if surr is None:
                req_up = copy.deepcopy(req)
                req_dn = copy.deepcopy(req)
                active_k = int(np.argmax(np.abs(got_rho)))
                req_up.Model.MarketDataSet.RatesCurvesBundles[curve_id].RatesSchedule.Rates[
                    active_k
                ] += h_rate
                req_dn.Model.MarketDataSet.RatesCurvesBundles[curve_id].RatesSchedule.Rates[
                    active_k
                ] -= h_rate
                pv_req_up = float(evaluate_differentiable_pv(req_up).item())
                pv_req_dn = float(evaluate_differentiable_pv(req_dn).item())
                fd_req_rho_k = (pv_req_up - pv_req_dn) / (2.0 * h_rate)
                assert abs(got_rho[active_k] - fd_req_rho_k) / max(abs(fd_req_rho_k), 1e-10) <= 1e-7

            # 3. DividendsSensitivities vs Central Finite Differences on discrete dividend amounts
            h_div = 1e-1 if surr is surrogate_f32 else 1e-4
            div_rel_thresh = 1e-4 if surr is surrogate_f32 else 1e-6
            for ticker in sorted_ids:
                got_div = res.dividends_arrays[ticker]
                if len(got_div) == 0:
                    continue
                base_div_t = state.dividend_schedule.dividend_amount_tensors[ticker].detach().clone()
                fd_div = np.zeros(len(got_div), dtype=np.float64)
                for k_ev in range(len(got_div)):
                    d_up = base_div_t.clone()
                    d_dn = base_div_t.clone()
                    d_up[k_ev] += h_div
                    d_dn[k_ev] -= h_div
                    pv_up = float(
                        evaluate_differentiable_pv(
                            state, div_amounts={ticker: d_up}, surrogate=surr
                        ).item()
                    )
                    pv_dn = float(
                        evaluate_differentiable_pv(
                            state, div_amounts={ticker: d_dn}, surrogate=surr
                        ).item()
                    )
                    fd_div[k_ev] = (pv_up - pv_dn) / (2.0 * h_div)

                norm_fd = float(np.linalg.norm(fd_div))
                if norm_fd > div_rel_thresh:
                    rel_div = _rel_err(got_div, fd_div)
                    assert rel_div <= 1e-5, f"DividendsSensitivities[{ticker}] rel err {rel_div:.3e} > 1e-5"
                    if surr is not surrogate_f32 and norm_fd > 1e-4:
                        assert rel_div <= 1e-7
                else:
                    tol_small = 2e-6 if surr is surrogate_f32 else 5e-8
                    assert float(np.linalg.norm(got_div - fd_div)) <= tol_small

            # 4. Vega & Cega vs Central Finite Differences (spot-checked on active asset/pair)
            if surr is not surrogate_f32:
                active_vega_ticker = max(
                    sorted_ids, key=lambda t: float(np.max(np.abs(res.vega_arrays[t])))
                )
                got_vega0 = res.vega_arrays[active_vega_ticker]
                e_idx, k_idx = np.unravel_index(int(np.argmax(np.abs(got_vega0))), got_vega0.shape)
                h_vol = 2e-7 if (surr is not None and abs(float(got_vega0[e_idx, k_idx])) > 1e-3) else 1e-6
                v_up = torch.zeros_like(torch.as_tensor(got_vega0))
                v_dn = torch.zeros_like(torch.as_tensor(got_vega0))
                v_up[e_idx, k_idx] = h_vol
                v_dn[e_idx, k_idx] = -h_vol
                pv_v_up = float(
                    evaluate_differentiable_pv(
                        state, vol_quote_deltas={active_vega_ticker: v_up}, surrogate=surr
                    ).item()
                )
                pv_v_dn = float(
                    evaluate_differentiable_pv(
                        state, vol_quote_deltas={active_vega_ticker: v_dn}, surrogate=surr
                    ).item()
                )
                fd_vega_cell = (pv_v_up - pv_v_dn) / (2.0 * h_vol)
                assert abs(got_vega0[e_idx, k_idx] - fd_vega_cell) / max(abs(fd_vega_cell), 1e-10) <= 1e-6

                # Cega on active off-diagonal pair
                got_cega_sorted = res.cega_tensor.detach().cpu().numpy()[state.lex_perm][:, state.lex_perm]
                ci, cj = np.unravel_index(int(np.argmax(np.abs(got_cega_sorted))), got_cega_sorted.shape)
                if ci == cj:
                    ci, cj = 0, 1
                h_corr = 1e-5
                c_up = state.corr_matrix.detach().clone()
                c_dn = state.corr_matrix.detach().clone()
                c_up[ci, cj] += h_corr
                c_up[cj, ci] += h_corr
                c_dn[ci, cj] -= h_corr
                c_dn[cj, ci] -= h_corr
                pv_c_up = float(
                    evaluate_differentiable_pv(state, corr_matrix=c_up, surrogate=surr).item()
                )
                pv_c_dn = float(
                    evaluate_differentiable_pv(state, corr_matrix=c_dn, surrogate=surr).item()
                )
                fd_cega_ij = (pv_c_up - pv_c_dn) / (2.0 * h_corr)
                if abs(fd_cega_ij) > 1e-6:
                    assert abs(got_cega_sorted[ci, cj] - fd_cega_ij) / abs(fd_cega_ij) <= 1e-6
                else:
                    assert abs(got_cega_sorted[ci, cj] - fd_cega_ij) <= 1e-11


def test_G1_9_euler_cash_gamma_identity_and_w2_dividend_curvature(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Gate G1.9: Verify |Gamma_{cash,agg}^{Euler} - sum_{i,j} S_i S_j Gamma_{ij}| <= 1e-9 (W2)."""
    req_ref, _ = parsed_dag

    torch.manual_seed(108)
    surrogate = MaskedEQAutocallSurrogate(
        d_max=8, m_max=48, d_model=128, n_egno_layers=3, n_sched_layers=2, n_heads=4
    ).to(dtype=torch.float64).eval()

    for req in (req_ref, multi_period_request):
        state = preprocess_eq_autocall(req)
        for surr in (None, surrogate):
            # 1. Compute via full D x D Gamma matrix
            res_full = compute_eq_autocall_greeks(
                state,
                queries=["Gamma", "MultivariateAggregatedCashGamma"],
                surrogate=surr,
            )
            # 2. Compute via O(1) 2-JVP directional Euler CashGamma without full Gamma
            res_euler = compute_eq_autocall_greeks(
                state,
                queries=["MultivariateAggregatedCashGamma"],
                surrogate=surr,
            )
            assert res_full.gamma_tensor is not None
            assert res_full.multivariate_aggregated_cash_gamma is not None
            assert res_euler.multivariate_aggregated_cash_gamma is not None
            assert res_euler.gamma is None

            # Verify quadratic form S_orig^T * Gamma_orig * S_orig
            inv_idx = torch.as_tensor(state.inv_lex_perm, dtype=torch.long)
            spots_orig = state.spots_t0.detach().index_select(0, inv_idx)
            quad_form = float(
                (spots_orig.unsqueeze(1) * spots_orig.unsqueeze(0) * res_full.gamma_tensor)
                .sum()
                .item()
            )
            euler_val = res_euler.multivariate_aggregated_cash_gamma

            diff_euler = abs(euler_val - quad_form)
            assert diff_euler <= 1e-9, (
                f"Gate G1.9 violation: |Euler CashGamma ({euler_val}) - S^T Gamma S ({quad_form})| = {diff_euler:.3e} > 1e-9"
            )
            assert abs(res_full.multivariate_aggregated_cash_gamma - quad_form) <= 1e-12

            # Verify against central finite difference of directional first derivative dPV/deps at S_0 * (1 +/- eps)
            h_eps = 1e-5
            base_s = state.spots_t0.detach()

            def _dir_slope(scale: float) -> float:
                s_in = (base_s * scale).detach().clone().requires_grad_(True)
                pv_s = evaluate_differentiable_pv(state, spots_t0=s_in, surrogate=surr)
                grad_s = torch.autograd.grad(pv_s, s_in)[0]
                return float((base_s * grad_s).sum().item())

            fd_cash_gamma = (_dir_slope(1.0 + h_eps) - _dir_slope(1.0 - h_eps)) / (2.0 * h_eps)
            assert abs(euler_val - fd_cash_gamma) <= max(1e-5 * abs(fd_cash_gamma), 1e-7)


def test_unpermutation_equivariance_across_all_8_greeks(
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Verify that permuting UnderlyingIdentifiers leaves scalar Greeks invariant and unpermutes per-asset Greeks."""
    req_orig = multi_period_request
    req_perm = copy.deepcopy(req_orig)

    perm = [3, 1, 4, 0, 2]
    orig_ids = list(req_orig.Product.UnderlyingIdentifiers)
    orig_refs = list(req_orig.Product.UnderlyingReferenceSpots)
    req_perm.Product.UnderlyingIdentifiers = [orig_ids[i] for i in perm]
    req_perm.Product.UnderlyingReferenceSpots = [orig_refs[i] for i in perm]

    all_queries = [
        "ProductValue",
        "CashFlows",
        "Delta",
        "Gamma",
        "MultivariateAggregatedCashGamma",
        "Vega",
        "Rho",
        "DividendsSensitivities",
        "Theta",
        "Cega",
    ]

    torch.manual_seed(77)
    surr = MaskedEQAutocallSurrogate().eval()

    g_orig: AutocallGreeksResult = compute_eq_autocall_greeks(
        req_orig, queries=all_queries, surrogate=surr
    )
    g_perm: AutocallGreeksResult = compute_eq_autocall_greeks(
        req_perm, queries=all_queries, surrogate=surr
    )

    assert g_orig.product_value == pytest.approx(g_perm.product_value, abs=1e-12)
    assert g_orig.multivariate_aggregated_cash_gamma == pytest.approx(
        g_perm.multivariate_aggregated_cash_gamma, abs=1e-12
    )
    assert g_orig.theta == pytest.approx(g_perm.theta, abs=1e-12)
    assert g_orig.rho == g_perm.rho

    assert g_orig.delta is not None and g_perm.delta is not None
    assert g_orig.gamma is not None and g_perm.gamma is not None
    assert g_orig.vega is not None and g_perm.vega is not None
    assert g_orig.dividends_sensitivities is not None and g_perm.dividends_sensitivities is not None
    assert g_orig.cega is not None and g_perm.cega is not None

    # Dict keys must appear in each request's respective original_identifiers order
    assert list(g_orig.delta.keys()) == orig_ids
    assert list(g_perm.delta.keys()) == req_perm.Product.UnderlyingIdentifiers

    for uid in orig_ids:
        assert g_orig.delta[uid] == pytest.approx(g_perm.delta[uid], abs=1e-12)
        np.testing.assert_allclose(g_orig.vega[uid], g_perm.vega[uid], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            g_orig.dividends_sensitivities[uid],
            g_perm.dividends_sensitivities[uid],
            rtol=1e-12,
            atol=1e-12,
        )
        for uid_j in orig_ids:
            assert g_orig.gamma[uid][uid_j] == pytest.approx(g_perm.gamma[uid][uid_j], abs=1e-12)
            assert g_orig.cega[uid][uid_j] == pytest.approx(g_perm.cega[uid][uid_j], abs=1e-12)
        # Diagonal of Cega must be 0.0
        assert g_orig.cega[uid][uid] == 0.0

    # Tensor attributes must follow the caller's permutation order
    assert g_orig.delta_tensor is not None and g_perm.delta_tensor is not None
    assert g_orig.gamma_tensor is not None and g_perm.gamma_tensor is not None
    assert g_orig.cega_tensor is not None and g_perm.cega_tensor is not None
    perm_t = torch.tensor(perm, dtype=torch.long)
    torch.testing.assert_close(g_perm.delta_tensor, g_orig.delta_tensor[perm_t], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        g_perm.gamma_tensor,
        g_orig.gamma_tensor[perm_t][:, perm_t],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        g_perm.cega_tensor,
        g_orig.cega_tensor[perm_t][:, perm_t],
        rtol=1e-12,
        atol=1e-12,
    )


def test_batched_surrogate_greeks_and_precision_segregation(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify compute_batched_surrogate_greeks precision segregation and Gate G2.4 CUDA throughput."""
    req, _ = parsed_dag
    state = preprocess_eq_autocall(req)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = MaskedEQAutocallSurrogate().to(device=device).eval()

    batch_size = 1024 if device.type == "cuda" else 64
    node_b = state.node_feats.to(device=device).expand(batch_size, -1, -1).contiguous()
    edge_b = state.edge_feats.to(device=device).expand(batch_size, -1, -1, -1).contiguous()
    amask_b = state.asset_mask.to(device=device).expand(batch_size, -1).contiguous()
    sched_b = state.sched_feats.to(device=device).expand(batch_size, -1, -1).contiguous()
    smask_b = state.sched_mask.to(device=device).expand(batch_size, -1).contiguous()
    glob_b = state.global_feats.to(device=device).expand(batch_size, -1).contiguous()

    # Warmup both compute_cash_gamma=True and compute_cash_gamma=False (2 calls so CUDAGraph captures before timing)
    _ = compute_batched_surrogate_greeks(
        surrogate, node_b, edge_b, amask_b, sched_b, smask_b, glob_b, compute_cash_gamma=True
    )
    for _ in range(2):
        _ = compute_batched_surrogate_greeks(
            surrogate, node_b, edge_b, amask_b, sched_b, smask_b, glob_b, compute_cash_gamma=False
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    batch_out = compute_batched_surrogate_greeks(
        surrogate, node_b, edge_b, amask_b, sched_b, smask_b, glob_b
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_trade = elapsed_ms / batch_size

    assert batch_out.product_value.dtype == torch.float64
    assert batch_out.delta.dtype == torch.float64
    assert batch_out.vega_atm.dtype == torch.float64
    assert batch_out.vega_surface_feats.dtype == torch.float64
    assert batch_out.rho_df.dtype == torch.float64
    assert batch_out.cega.dtype == torch.float64
    assert torch.isfinite(batch_out.product_value).all()
    assert torch.isfinite(batch_out.delta).all()
    assert torch.isfinite(batch_out.cega).all()

    # Inactive asset slots (5..7) and inactive schedule slots (1..47) must have strictly 0.0 Greeks
    assert batch_out.delta[:, 5:].abs().max().item() == 0.0
    assert batch_out.vega_atm[:, 5:, :].abs().max().item() == 0.0
    assert batch_out.rho_df[:, 1:].abs().max().item() == 0.0

    if device.type == "cuda" and sys.gettrace() is None:
        assert ms_per_trade <= 0.05, (
            f"Gate G2.4 violation: batched GPU Greeks took {ms_per_trade:.4f} ms/trade > 0.05 ms/trade"
        )


def test_greeks_engine_m5_branches_modifiers_qmc_crn_and_batch_variants(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Covers boundary casts, #112 m_rem==0/already_autocalled, modifiers, dual-curve Rho, QMC CRN, and batched variants."""
    from deepvol.api.schemas_eq_autocall import FixingsSpec, MonitoringIndexEnum
    from deepvol.calibration.autocall_sobol_generator import generate_synthetic_autocall_batch
    from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig
    from deepvol.models.autocall.greeks_engine import (
        cast_from_surrogate_boundary,
        cast_to_surrogate_boundary,
        compute_qmc_crn_greeks,
    )

    req_ref, _ = parsed_dag
    state_ref = preprocess_eq_autocall(req_ref)
    surr_cpu = MaskedEQAutocallSurrogate().eval()

    # 1. Surrogate boundary cast helpers (float32 <-> float64 with gradients)
    cpu_dev = torch.device("cpu")
    x_64 = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    x_32 = cast_to_surrogate_boundary(x_64, device=cpu_dev, dtype=torch.float32)
    assert x_32.dtype == torch.float32
    y_64 = cast_from_surrogate_boundary(x_32 * 2.0, device=cpu_dev)
    assert y_64.dtype == torch.float64
    (g_x64,) = torch.autograd.grad(y_64.sum(), (x_64,))
    assert torch.allclose(g_x64, torch.full_like(x_64, 2.0))

    # 2. Regression #112: settled (m_rem == 0) and already_autocalled=True return finite Greeks & zero Vega
    req_m0 = copy.deepcopy(multi_period_request)
    as_of_m0 = req_m0.Product.ObservationDates[-1] + datetime.timedelta(days=30)
    req_m0.Model.MarketDataSet.AsOfDate = as_of_m0
    req_m0.Product.PaymentDates = [
        d + datetime.timedelta(days=3) for d in req_m0.Product.ObservationDates
    ]
    req_m0.Model.MarketDataSet.Fixings = {
        uid: FixingsSpec(
            Identifier=uid,
            Dates=list(req_m0.Product.ObservationDates),
            Values=[90.0] * len(req_m0.Product.ObservationDates),
        )
        for uid in req_m0.Product.UnderlyingIdentifiers
    }
    st_m0 = preprocess_eq_autocall(req_m0)
    assert st_m0.m_rem == 0
    for surr_use in (None, surr_cpu):
        g_m0 = compute_eq_autocall_greeks(
            st_m0,
            queries=["ProductValue", "Delta", "Gamma", "MultivariateAggregatedCashGamma", "Vega", "Rho", "DividendsSensitivities", "Theta", "Cega"],
            surrogate=surr_use,
        )
        assert np.isfinite(g_m0.product_value)
        assert g_m0.vega_arrays is not None and all(np.all(v == 0.0) for v in g_m0.vega_arrays.values())

    req_ko = copy.deepcopy(multi_period_request)
    st_ko = preprocess_eq_autocall(req_ko)
    st_ko.already_autocalled = True
    st_ko.vol_surfaces = {}
    g_ko = compute_eq_autocall_greeks(
        st_ko,
        queries=["ProductValue", "Delta", "Vega", "Rho", "Theta", "Cega"],
        surrogate=None,
    )
    assert np.isfinite(g_ko.product_value)
    assert g_ko.vega_arrays is not None and all(np.all(v == 0.0) for v in g_ko.vega_arrays.values())

    # 3. Exotic modifiers + dual-curve funding + RateFixings + BestOf/AverageOf + ConvexCega/CashGamma-only
    req_ex = copy.deepcopy(multi_period_request)
    req_ex.Product.CallabilityMonitoringIndex = MonitoringIndexEnum.BestOf
    req_ex.Product.CouponBarrierMonitoringIndex = MonitoringIndexEnum.AverageOf
    req_ex.Product.OptionMonitoringIndex = MonitoringIndexEnum.WorstOf
    req_ex.Product.CouponBarrierType = "Out"
    req_ex.Product.CouponBarrierDirection = "Down"
    req_ex.Product.LowerStrike = 0.55
    req_ex.Product.AutoCallCoupon = 0.02
    req_ex.Product.ForceCouponsOnAutocall = True
    req_ex.Product.NoNotionalRepayment = True
    req_ex.Product.OneStarStrike = 1.15
    req_ex.Product.OneStarCoupon = 0.03
    req_ex.Product.ParticipationCallStrike = 1.00
    req_ex.Product.ParticipationCallRate = 0.25
    req_ex.Product.AutoCallTriggerAbsoluteDigitalSpread = 0.0
    req_ex.Product.IsLongBasket = True
    req_ex.Product.RateIdentifier = "RUB_MOSPRIME_3M"
    req_ex.Product.FundingSpread = 0.015
    req_ex.Product.FundingLegPaymentDates = list(req_ex.Product.PaymentDates)
    base_cid = next(iter(req_ex.Model.MarketDataSet.RatesCurvesBundles))
    mosprime_c = copy.deepcopy(req_ex.Model.MarketDataSet.RatesCurvesBundles[base_cid])
    mosprime_c.Identifier = "RUB_MOSPRIME_3M"
    req_ex.Model.MarketDataSet.RatesCurvesBundles["RUB_MOSPRIME_3M"] = mosprime_c
    st_ex = preprocess_eq_autocall(req_ex)
    g_ex = compute_eq_autocall_greeks(
        st_ex,
        queries=["ProductValue", "Delta", "MultivariateAggregatedCashGamma", "Vega", "Rho", "DividendsSensitivities", "Theta", "ConvexCega"],
        surrogate=surr_cpu,
    )
    assert np.isfinite(g_ex.product_value) and g_ex.multivariate_aggregated_cash_gamma is not None and g_ex.cega is not None

    # ProductValue-only & CashGamma-only single-trade fast queries
    g_pv_only = compute_eq_autocall_greeks(state_ref, queries=["ProductValue"], surrogate=surr_cpu)
    assert np.isfinite(g_pv_only.product_value) and g_pv_only.delta is None
    g_cg_only = compute_eq_autocall_greeks(state_ref, queries=["ProductValue", "MultivariateAggregatedCashGamma"], surrogate=None)
    assert g_cg_only.multivariate_aggregated_cash_gamma is not None and g_cg_only.gamma is None

    # 4. QMC CRN Greeks (full 8 Greeks + Euler CashGamma-only + m_rem==0 settled)
    req_qmc = copy.deepcopy(req_ref)
    req_qmc.Product.UnderlyingIdentifiers = req_ref.Product.UnderlyingIdentifiers[:2]
    req_qmc.Product.UnderlyingReferenceSpots = req_ref.Product.UnderlyingReferenceSpots[:2]
    req_qmc.Product.PredefinedDividends = None
    tks2 = req_qmc.Product.UnderlyingIdentifiers
    req_qmc.Model.MarketDataSet.Spots = {k: req_ref.Model.MarketDataSet.Spots[k] for k in tks2}
    req_qmc.Model.MarketDataSet.VolatilitySurfaces = {
        k: req_ref.Model.MarketDataSet.VolatilitySurfaces[k] for k in tks2
    }
    req_qmc.Model.MarketDataSet.Dividends = {k: req_ref.Model.MarketDataSet.Dividends[k] for k in tks2}
    req_qmc.Model.MarketDataSet.Fixings = {
        k: req_ref.Model.MarketDataSet.Fixings[k]
        for k in tks2
        if k in req_ref.Model.MarketDataSet.Fixings
    }
    req_qmc.Model.MarketDataSet.Correlations = {f"{tks2[0]}&{tks2[1]}": 0.4}
    st_qmc = preprocess_eq_autocall(req_qmc)
    cfg_qmc = DupireQMCConfig(n_paths=128, n_replicates=2, n_substeps=2)
    qmc_all = compute_qmc_crn_greeks(
        st_qmc,
        queries=["ProductValue", "Delta", "Gamma", "MultivariateAggregatedCashGamma", "Vega", "Rho", "DividendsSensitivities", "Theta", "Cega"],
        config=cfg_qmc,
    )
    assert np.isfinite(qmc_all.product_value) and qmc_all.delta is not None and qmc_all.gamma is not None
    assert qmc_all.cega is not None and qmc_all.theta is not None
    qmc_cg = compute_qmc_crn_greeks(
        st_qmc,
        queries=["ProductValue", "MultivariateAggregatedCashGamma"],
        config=cfg_qmc,
    )
    assert qmc_cg.multivariate_aggregated_cash_gamma is not None and qmc_cg.gamma is None
    qmc_m0 = compute_qmc_crn_greeks(st_m0, queries=["ProductValue", "Delta", "Vega"], config=cfg_qmc)
    assert qmc_m0.vega_arrays is not None and all(np.all(v == 0.0) for v in qmc_m0.vega_arrays.values())

    # 5. Batched surrogate Greeks across D<=5 small batch (cg=True fastpath), D=8 dynamic batch w/ modifiers & f64 CPU
    synth_b = generate_synthetic_autocall_batch(batch_size=8, d_max=8, m_max=24, n_mc_paths=32, seed=77, device="cpu", dtype=torch.float64)
    surr_d8 = MaskedEQAutocallSurrogate(d_max=8, m_max=24).eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surr_d8_dev = copy.deepcopy(surr_d8).to(device=dev)
    b8 = 8
    out_d8 = compute_batched_surrogate_greeks(
        surr_d8_dev,
        synth_b.node_feats.to(device=dev),
        synth_b.edge_feats.to(device=dev),
        synth_b.asset_mask.to(device=dev),
        synth_b.sched_feats.to(device=dev),
        synth_b.sched_mask.to(device=dev),
        synth_b.global_feats.to(device=dev),
        spots_t0=torch.full((b8, 8), 100.0, dtype=torch.float64, device=dev),
        ref_spots=torch.full((b8, 8), 100.0, dtype=torch.float64, device=dev),
        pv_in_transit=torch.zeros((b8,), dtype=torch.float64, device=dev),
        no_notional_repayment=True,
        autocall_coupon=0.02,
        force_coupons_on_autocall=True,
        is_long_basket=torch.ones((b8,), dtype=torch.bool, device=dev),
        w_fund_per_period=torch.full((b8, 24), 0.001, dtype=torch.float64, device=dev),
        lower_strike=torch.full((b8,), 0.5, dtype=torch.float64, device=dev),
        one_star_pv=0.01,
        part_call_pv=0.02,
        compute_cash_gamma=True,
        require_second_order_sdpa=True,
    )
    assert torch.isfinite(out_d8.product_value).all()
    assert out_d8.cash_gamma_euler is not None and torch.isfinite(out_d8.cash_gamma_euler).all()

    # D=5 small batch B=4 cg=True + float64 CPU surrogate path
    t2_ref = state_ref.to_tier2_float32_tensors()
    surr_d5 = MaskedEQAutocallSurrogate().to(device=dev).eval()
    out_d5_small = compute_batched_surrogate_greeks(
        surr_d5,
        t2_ref["node_feats"].to(device=dev).expand(4, -1, -1).contiguous(),
        t2_ref["edge_feats"].to(device=dev).expand(4, -1, -1, -1).contiguous(),
        t2_ref["asset_mask"].to(device=dev).expand(4, -1).contiguous(),
        t2_ref["sched_feats"].to(device=dev).expand(4, -1, -1).contiguous(),
        t2_ref["sched_mask"].to(device=dev).expand(4, -1).contiguous(),
        t2_ref["global_feats"].to(device=dev).expand(4, -1).contiguous(),
        compute_cash_gamma=True,
    )
    assert out_d5_small.cash_gamma_euler is not None and torch.isfinite(out_d5_small.cash_gamma_euler).all()

    surr_f64_cpu = MaskedEQAutocallSurrogate(d_max=8, m_max=24).to(dtype=torch.float64).eval()
    out_cpu64 = compute_batched_surrogate_greeks(
        surr_f64_cpu,
        synth_b.node_feats[:2],
        synth_b.edge_feats[:2],
        synth_b.asset_mask[:2],
        synth_b.sched_feats[:2],
        synth_b.sched_mask[:2],
        synth_b.global_feats[:2],
        compute_cash_gamma=True,
    )
    assert out_cpu64.cash_gamma_euler is not None and torch.isfinite(out_cpu64.cash_gamma_euler).all()


def test_batched_surrogate_greeks_padded_mask_regression_140_142() -> None:
    """Locks down #140/#142: padded sched_mask (m_rem < m_max) & asset_mask across Markov and non-Markov paths."""
    from deepvol.calibration.autocall_sobol_generator import generate_synthetic_autocall_batch
    from deepvol.models.autocall.feature_layout import AutocallGlobalCol, MonitoringIndexCode

    synth_mix = generate_synthetic_autocall_batch(
        batch_size=16, d_max=8, m_max=48, n_mc_paths=64, seed=202610, device="cpu", dtype=torch.float64
    )
    surr_cpu = MaskedEQAutocallSurrogate().eval()
    devs: list[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))

    for dev in devs:
        surr_dev = copy.deepcopy(surr_cpu).to(device=dev).eval()
        # 1. Mixed call_index (use_markov_tier1=False) with m_rem < 48 & D < 8
        for cg in (False, True):
            out_m = compute_batched_surrogate_greeks(
                surr_dev,
                synth_mix.node_feats.to(device=dev),
                synth_mix.edge_feats.to(device=dev),
                synth_mix.asset_mask.to(device=dev),
                synth_mix.sched_feats.to(device=dev),
                synth_mix.sched_mask.to(device=dev),
                synth_mix.global_feats.to(device=dev),
                compute_cash_gamma=cg,
            )
            for fname in ("product_value", "delta", "vega_atm", "vega_surface_feats", "rho_df", "cega"):
                val = getattr(out_m, fname)
                assert torch.isfinite(val).all(), f"{dev} mix cg={cg} non-finite {fname}"
            if cg:
                assert out_m.cash_gamma_euler is not None and torch.isfinite(out_m.cash_gamma_euler).all()

        # 2. Homogeneous WorstOf & BestOf (use_markov_tier1=True) with mixed D in {2,4,5,7,8} & m_rem in {1,3,12,48}
        for idx_code in (MonitoringIndexCode.WORST_OF, MonitoringIndexCode.BEST_OF):
            b_sz = 8
            node_h = synth_mix.node_feats[:b_sz].clone()
            edge_h = synth_mix.edge_feats[:b_sz].clone()
            amask_h = torch.zeros((b_sz, 8), dtype=torch.float64)
            sched_h = synth_mix.sched_feats[:b_sz].clone()
            smask_h = torch.zeros((b_sz, 48), dtype=torch.float64)
            glob_h = synth_mix.global_feats[:b_sz].clone()
            glob_h[:, int(AutocallGlobalCol.CALL_INDEX_ID)] = float(idx_code)
            glob_h[:, int(AutocallGlobalCol.CPN_INDEX_ID)] = float(idx_code)
            glob_h[:, int(AutocallGlobalCol.OPT_INDEX_ID)] = float(idx_code)
            d_list = (2, 4, 5, 7, 8, 2, 5, 8)
            m_list = (1, 3, 12, 48, 1, 3, 12, 48)
            for b_i, (d_i, m_i) in enumerate(zip(d_list, m_list)):
                amask_h[b_i, :d_i] = 1.0
                node_h[b_i, d_i:, :] = 0.0
                smask_h[b_i, :m_i] = 1.0
                sched_h[b_i, m_i:, :] = 0.0

            for cg in (False, True):
                out_h = compute_batched_surrogate_greeks(
                    surr_dev,
                    node_h.to(device=dev),
                    edge_h.to(device=dev),
                    amask_h.to(device=dev),
                    sched_h.to(device=dev),
                    smask_h.to(device=dev),
                    glob_h.to(device=dev),
                    compute_cash_gamma=cg,
                )
                for fname in ("product_value", "delta", "vega_atm", "vega_surface_feats", "rho_df", "cega"):
                    val = getattr(out_h, fname)
                    assert torch.isfinite(val).all(), f"{dev} idx={idx_code} cg={cg} non-finite {fname}"
                if cg:
                    assert out_h.cash_gamma_euler is not None and torch.isfinite(out_h.cash_gamma_euler).all()
                am_bool = amask_h > 0.5
                sm_bool = smask_h > 0.5
                assert float(out_h.delta.cpu()[~am_bool].abs().max()) == 0.0
                assert float(out_h.vega_atm.cpu().view(b_sz, 8, 2)[~am_bool].abs().max()) == 0.0
                assert float(out_h.rho_df.cpu()[~sm_bool].abs().max()) == 0.0


def test_greeks_audit_blind_spots(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    from deepvol.api.schemas_eq_autocall import (
        CurveInterpolationMethodEnum,
        DayCountConventionEnum,
        EquityVolSmileCollectionSpec,
        InterpolationSpaceCoordinatesEnum,
        RateQuoteExpressionEnum,
    )
    from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig
    from deepvol.models.autocall.feature_layout import AutocallNodeCol
    from deepvol.models.autocall.greeks_engine import compute_qmc_crn_greeks

    req_ref, _ = parsed_dag

    # LogarithmOfDiscountFactors and DiscountFactors Rho must match central FD (negative sign).
    for space_c, interp_m in (
        (InterpolationSpaceCoordinatesEnum.LogarithmOfDiscountFactors, CurveInterpolationMethodEnum.LogLinear),
        (InterpolationSpaceCoordinatesEnum.DiscountFactors, CurveInterpolationMethodEnum.PiecewiseConstantForward),
        (InterpolationSpaceCoordinatesEnum.LogarithmOfDiscountFactors, CurveInterpolationMethodEnum.Linear),
    ):
        req_sp = copy.deepcopy(multi_period_request)
        cid = next(iter(req_sp.Model.MarketDataSet.RatesCurvesBundles))
        bundle = req_sp.Model.MarketDataSet.RatesCurvesBundles[cid]
        bundle.InterpolationMethod = interp_m
        bundle.InterpolationSpaceCoordinates = space_c
        bundle.RateQuoteExpression = RateQuoteExpressionEnum.Decimal
        as_of_sp = bundle.AsOfDate
        tenors_yr = np.array(
            [(d - as_of_sp).total_seconds() / (365.0 * 86400.0) for d in bundle.RatesSchedule.Dates],
            dtype=np.float64,
        )
        r_dec = np.asarray(bundle.RatesSchedule.Rates, dtype=np.float64) / 100.0
        log_df = -r_dec * tenors_yr
        bundle.RatesSchedule.Rates = (
            log_df.tolist()
            if space_c == InterpolationSpaceCoordinatesEnum.LogarithmOfDiscountFactors
            else np.exp(log_df).tolist()
        )

        st_sp = preprocess_eq_autocall(req_sp)
        res_sp = compute_eq_autocall_greeks(st_sp, queries=["ProductValue", "Rho"])
        assert res_sp.rho_arrays is not None
        got_rho_sp = res_sp.rho_arrays[cid]
        fd_rho_sp = np.zeros_like(got_rho_sp)
        h_q = 1e-6
        for k in range(len(got_rho_sp)):
            r_u = copy.deepcopy(req_sp)
            r_d = copy.deepcopy(req_sp)
            r_u.Model.MarketDataSet.RatesCurvesBundles[cid].RatesSchedule.Rates[k] += h_q
            r_d.Model.MarketDataSet.RatesCurvesBundles[cid].RatesSchedule.Rates[k] -= h_q
            out_u = evaluate_differentiable_pv(r_u)
            out_d = evaluate_differentiable_pv(r_d)
            assert isinstance(out_u, torch.Tensor) and isinstance(out_d, torch.Tensor)
            fd_rho_sp[k] = (float(out_u.item()) - float(out_d.item())) / (2.0 * h_q)
        assert _rel_err(got_rho_sp, fd_rho_sp) <= 1e-5

    # Batched non-Markov Tier-1 Cega must NOT double-count symmetric off-diagonals.
    surr_zero = MaskedEQAutocallSurrogate(
        d_max=8, m_max=48, d_model=64, n_egno_layers=1, n_sched_layers=1
    ).to(dtype=torch.float64).eval()
    surr_zero.zero_init_residual_heads(std=1e-3)
    assert not surr_zero.is_zero_residual()
    req_avg = copy.deepcopy(req_ref)
    req_avg.Product.CallabilityMonitoringIndex = MonitoringIndexEnum.AverageOf
    req_avg.Product.CouponBarrierMonitoringIndex = MonitoringIndexEnum.AverageOf
    req_avg.Product.OptionMonitoringIndex = MonitoringIndexEnum.AverageOf
    st_avg = preprocess_eq_autocall(req_avg)
    out_cega = compute_batched_surrogate_greeks(
        surr_zero,
        st_avg.node_feats.to(dtype=torch.float64),
        st_avg.edge_feats.to(dtype=torch.float64),
        st_avg.asset_mask.to(dtype=torch.float64),
        st_avg.sched_feats.to(dtype=torch.float64),
        st_avg.sched_mask.to(dtype=torch.float64),
        st_avg.global_feats.to(dtype=torch.float64),
        spots_t0=st_avg.spots_t0.unsqueeze(0),
        ref_spots=st_avg.ref_spots.unsqueeze(0),
        pv_in_transit=st_avg.pv_in_transit,
        no_notional_repayment=st_avg.no_notional_repayment,
        autocall_coupon=st_avg.autocall_coupon,
        force_coupons_on_autocall=st_avg.force_coupons_on_autocall,
        is_long_basket=st_avg.is_long_basket,
        lower_strike=st_avg.lower_strike,
        forward_perf=st_avg.forward_perf.unsqueeze(0),
        v_perf=st_avg.v_perf.unsqueeze(0),
        cash_div_leverage=st_avg.cash_div_leverage.unsqueeze(0),
        compute_cash_gamma=False,
    )
    h_c = 1e-5
    c_u, c_d = st_avg.corr_matrix.clone(), st_avg.corr_matrix.clone()
    c_u[0, 1] += h_c
    c_u[1, 0] += h_c
    c_d[0, 1] -= h_c
    c_d[1, 0] -= h_c
    pv_cu_t = evaluate_differentiable_pv(st_avg, corr_matrix=c_u, surrogate=surr_zero)
    pv_cd_t = evaluate_differentiable_pv(st_avg, corr_matrix=c_d, surrogate=surr_zero)
    assert isinstance(pv_cu_t, torch.Tensor) and isinstance(pv_cd_t, torch.Tensor)
    pv_cu = float(pv_cu_t.item())
    pv_cd = float(pv_cd_t.item())
    fd_cega_01 = (pv_cu - pv_cd) / (2.0 * h_c)
    assert abs(float(out_cega.cega[0, 0, 1].item()) - fd_cega_01) <= 1e-4

    # Deep-ITM KO erfcx stability (u_hi < -4.0) across float64 AD.
    st_mp = preprocess_eq_autocall(multi_period_request)
    nf_b = st_mp.node_feats.to(dtype=torch.float64)
    ef_b = st_mp.edge_feats.to(dtype=torch.float64)
    am_b = st_mp.asset_mask.to(dtype=torch.float64)
    sf_b = st_mp.sched_feats.to(dtype=torch.float64)
    sm_b = st_mp.sched_mask.to(dtype=torch.float64)
    gf_b = st_mp.global_feats.to(dtype=torch.float64).clone()
    nf_itm = nf_b.clone()
    nf_itm[0, : st_mp.d_assets, int(AutocallNodeCol.LOG_MONEYNESS)] = -2.5
    out_itm = compute_batched_surrogate_greeks(
        surr_zero, nf_itm, ef_b, am_b, sf_b, sm_b, gf_b, compute_cash_gamma=True
    )
    assert torch.isfinite(out_itm.product_value).all()
    assert torch.isfinite(out_itm.delta).all()
    assert out_itm.cash_gamma_euler is not None and torch.isfinite(out_itm.cash_gamma_euler).all()

    # Standalone MultivariateAggregatedCashGamma in QMC CRN Greeks must include cross-gamma.
    tks2 = req_ref.Product.UnderlyingIdentifiers[:2]
    req2 = copy.deepcopy(multi_period_request)
    req2.Product.UnderlyingIdentifiers = tks2
    req2.Product.UnderlyingReferenceSpots = req2.Product.UnderlyingReferenceSpots[:2]
    req2.Model.MarketDataSet.Spots = {k: req2.Model.MarketDataSet.Spots[k] for k in tks2}
    req2.Model.MarketDataSet.VolatilitySurfaces = {
        k: req2.Model.MarketDataSet.VolatilitySurfaces[k] for k in tks2
    }
    surf0 = req2.Model.MarketDataSet.VolatilitySurfaces[tks2[0]]
    assert isinstance(surf0, EquityVolSmileCollectionSpec)
    surf0.DayCountConvention = DayCountConventionEnum.Act360
    surf0.SmileInterpolationMethod = CurveInterpolationMethodEnum.PCHIP
    req2.Model.MarketDataSet.Dividends = {
        k: req2.Model.MarketDataSet.Dividends[k]
        for k in tks2
        if k in req2.Model.MarketDataSet.Dividends
    }
    req2.Product.PredefinedDividends = {}
    req2.Model.MarketDataSet.Fixings = {
        k: req2.Model.MarketDataSet.Fixings[k] for k in tks2 if k in req2.Model.MarketDataSet.Fixings
    }
    req2.Model.MarketDataSet.Correlations = {f"{tks2[0]}&{tks2[1]}": 0.45}
    st2 = preprocess_eq_autocall(req2)
    cfg2 = DupireQMCConfig(n_paths=256, n_replicates=4, n_substeps=2)
    res_full = compute_qmc_crn_greeks(
        st2,
        queries=["ProductValue", "Gamma", "MultivariateAggregatedCashGamma", "Cega", "Vega", "Theta"],
        config=cfg2,
    )
    res_cg_only = compute_qmc_crn_greeks(
        st2,
        queries=["ProductValue", "MultivariateAggregatedCashGamma"],
        config=cfg2,
    )
    assert res_full.multivariate_aggregated_cash_gamma is not None
    assert res_cg_only.multivariate_aggregated_cash_gamma is not None
    assert float(res_cg_only.multivariate_aggregated_cash_gamma) == pytest.approx(
        float(res_full.multivariate_aggregated_cash_gamma), abs=1e-4, rel=1e-2
    )
    assert res_full.vega_arrays is not None and res_full.cega is not None and res_full.theta is not None

    # QMC CRN Cega vs central correlation bump and FundingLeg Theta + In-Transit m_rem==0.
    req_fl = copy.deepcopy(req2)
    req_fl.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_fl.Product.FundingSpread = 0.02
    req_fl.Product.FundingLegPaymentDates = list(req_fl.Product.PaymentDates)
    as_of_fl = req_fl.Model.MarketDataSet.AsOfDate
    req_fl.Product.PaymentDates[-2] = as_of_fl + datetime.timedelta(days=1)
    req_fl.Product.FundingLegPaymentDates[-2] = as_of_fl + datetime.timedelta(days=1)
    st_fl_theta = preprocess_eq_autocall(req_fl)
    res_fl_th = compute_qmc_crn_greeks(
        st_fl_theta, queries=["ProductValue", "Theta", "Rho"], config=cfg2
    )
    assert res_fl_th.theta is not None and np.isfinite(float(res_fl_th.theta))

    # Factored Vega (e, k) finite-difference check on Act360 / PCHIP surface.
    cfg_vega = DupireQMCConfig(n_paths=2048, n_replicates=4, n_substeps=8)
    res_vega_pchip = compute_qmc_crn_greeks(
        st2, queries=["ProductValue", "Vega"], config=cfg_vega
    )
    assert res_vega_pchip.vega_arrays is not None
    v_mat = res_vega_pchip.vega_arrays[tks2[0]]
    ei, ki = np.unravel_index(int(np.argmax(np.abs(v_mat))), v_mat.shape)
    h_v = 1e-4
    r_vu, r_vd = copy.deepcopy(req2), copy.deepcopy(req2)
    surf_vu = r_vu.Model.MarketDataSet.VolatilitySurfaces[tks2[0]]
    surf_vd = r_vd.Model.MarketDataSet.VolatilitySurfaces[tks2[0]]
    assert isinstance(surf_vu, EquityVolSmileCollectionSpec)
    assert isinstance(surf_vd, EquityVolSmileCollectionSpec)
    surf_vu.VolatilityQuotes[ei][ki] += h_v
    surf_vd.VolatilityQuotes[ei][ki] -= h_v
    pv_vu = compute_qmc_crn_greeks(preprocess_eq_autocall(r_vu), queries=["ProductValue"], config=cfg_vega).product_value
    pv_vd = compute_qmc_crn_greeks(preprocess_eq_autocall(r_vd), queries=["ProductValue"], config=cfg_vega).product_value
    fd_v_ek = (float(pv_vu) - float(pv_vd)) / (2.0 * h_v)
    assert abs(float(v_mat[ei, ki]) - fd_v_ek) <= max(1e-3, 0.15 * abs(fd_v_ek))


def test_qmc_aad_v3_is_long_basket_and_onestar_parity(
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Verifies V3 memory VJP, IsLongBasket I_ramp VJP, and OneStar shifted left-spread on CPU."""
    from deepvol.models.autocall.qmc_aad_greeks import (
        QmcAadGreeksConfig,
        _compute_payoff_aad_state_adjoints_torch,
        compute_qmc_aad_crn_greeks,
    )

    # 1) V3 literal forward & VJP vs central FD across perf_base on overlapping spreads.
    rng = np.random.default_rng(123)
    m_rem, d_assets, n_tot = 3, 2, 256
    perf0 = torch.tensor(
        np.exp(0.18 * rng.standard_normal((m_rem, d_assets, n_tot))),
        dtype=torch.float64,
    )
    ko_b = torch.tensor([0.95, 0.92, 0.90], dtype=torch.float64)
    is_c = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    cpn_b = torch.tensor([0.78, 0.78, 0.78], dtype=torch.float64)
    is_g = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    cpn_r = torch.tensor([0.04, 0.04, 0.04], dtype=torch.float64)
    df_p = torch.tensor([0.96, 0.92, 0.88], dtype=torch.float64)
    wf = torch.tensor([0.01, 0.01, 0.01], dtype=torch.float64)
    tau_o = np.array([0.25, 0.50, 0.75], dtype=np.float64)

    s1_box: list[torch.Tensor] = []
    bar_lx, bar_ln_fwd, _ = _compute_payoff_aad_state_adjoints_torch(
        perf0,
        ko_barriers=ko_b,
        is_callable=is_c,
        cpn_barriers=cpn_b,
        is_guar=is_g,
        cpn_rates=cpn_r,
        df_pay=df_p,
        w_fund=wf,
        inv_perm=[0, 1],
        call_code=0,
        cpn_code=0,
        opt_code=0,
        ko_spread=-0.10,
        cpn_spread=-0.10,
        ki_spread=-0.10,
        cpn_dir_down=False,
        cpn_type_out=False,
        is_memory=True,
        force_cpn_ac=False,
        no_notional=False,
        is_long_basket=True,
        has_ki=True,
        ki_barrier=1.0,
        i_ki_past=0.0,
        ki_monitoring_type="European",
        tau_obs=tau_o,
        vol_atm_obs=None,
        strike=0.90,
        lower_strike=None,
        option_gearing=1.111111,
        ac_cpn=0.0,
        c_mem_0=0.04,
        part_call_strike=None,
        part_call_rate=0.0,
        one_star_strike=1.10,
        one_star_coupon=0.02,
        one_star_eps=0.10,
        barrier_eps_min=0.015,
        barrier_eps_max=0.02,
        n_rep=n_tot,
        emit_stage1=True,
        out_stage1_rep=s1_box,
    )
    assert len(s1_box) == 1
    h_fd = 1e-6
    p_u = perf0.clone()
    p_d = perf0.clone()
    p_u[2, 0, :] *= float(np.exp(h_fd))
    p_d[2, 0, :] *= float(np.exp(-h_fd))
    s_u: list[torch.Tensor] = []
    s_d: list[torch.Tensor] = []
    kw_common: dict[str, Any] = dict(
        ko_barriers=ko_b,
        is_callable=is_c,
        cpn_barriers=cpn_b,
        is_guar=is_g,
        cpn_rates=cpn_r,
        df_pay=df_p,
        w_fund=wf,
        inv_perm=[0, 1],
        call_code=0,
        cpn_code=0,
        opt_code=0,
        ko_spread=-0.10,
        cpn_spread=-0.10,
        ki_spread=-0.10,
        cpn_dir_down=False,
        cpn_type_out=False,
        is_memory=True,
        force_cpn_ac=False,
        no_notional=False,
        is_long_basket=True,
        has_ki=True,
        ki_barrier=1.0,
        i_ki_past=0.0,
        ki_monitoring_type="European",
        tau_obs=tau_o,
        vol_atm_obs=None,
        strike=0.90,
        lower_strike=None,
        option_gearing=1.111111,
        ac_cpn=0.0,
        c_mem_0=0.04,
        part_call_strike=None,
        part_call_rate=0.0,
        one_star_strike=1.10,
        one_star_coupon=0.02,
        one_star_eps=0.10,
        barrier_eps_min=0.015,
        barrier_eps_max=0.02,
        n_rep=n_tot,
        emit_stage1=True,
        compute_grads=False,
    )
    _compute_payoff_aad_state_adjoints_torch(p_u, out_stage1_rep=s_u, **kw_common)
    _compute_payoff_aad_state_adjoints_torch(p_d, out_stage1_rep=s_d, **kw_common)
    c0_s = 3 * m_rem
    fd_ln = (float(s_u[0][0, c0_s + 4]) - float(s_d[0][0, c0_s + 4])) / (2.0 * h_fd)
    assert abs(float(bar_ln_fwd[0, 2]) - fd_ln) <= 1e-4
    assert torch.isfinite(bar_lx).all()

    # 2) OneStar left-shifted spread [1.00, 1.10] when KI is active vs centered [1.09, 1.11] when KI is None.
    req_os = copy.deepcopy(multi_period_request)
    req_os.Product.OneStarStrike = 1.10
    req_os.Product.OneStarCoupon = 0.05
    req_os.Product.AutoCallTriggerAbsoluteDigitalSpread = -0.10
    req_os.Product.OptionDownAndInBarrierLevel = 0.78
    req_os.Product.OptionDownAndInBarrierAbsoluteDigitalSpread = -0.05
    st_ki = preprocess_eq_autocall(req_os)
    cfg_os = QmcAadGreeksConfig(n_paths=512, n_replicates=4, n_substeps=2)
    res_ki = compute_qmc_aad_crn_greeks(st_ki, req_os, queries=["ProductValue", "Delta"], config=cfg_os, device="cpu")
    req_no_ki = copy.deepcopy(req_os)
    req_no_ki.Product.OptionDownAndInBarrierLevel = None
    st_no_ki = preprocess_eq_autocall(req_no_ki)
    res_no_ki = compute_qmc_aad_crn_greeks(st_no_ki, req_no_ki, queries=["ProductValue"], config=cfg_os, device="cpu")
    assert np.isfinite(res_ki.product_value) and np.isfinite(res_no_ki.product_value)


def test_qmc_aad_murray_cholesky_4mode_ki_and_device_respect(
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Checks Murray (2016) lower-Cholesky Cega vs FD, 4-mode KI, MF-CV, and device/query invariance."""
    from deepvol.models.autocall.qmc_aad_greeks import (
        QmcAadGreeksConfig,
        cholesky_corr_adjoint_np,
        compute_qmc_aad_crn_greeks,
    )

    req_eq = copy.deepcopy(multi_period_request)
    # Use explicit smooth digital spreads so Stage-1 FD and Stage-2 AAD Cega share the exact smoothed payoff.
    req_eq.Product.AutoCallTriggerAbsoluteDigitalSpread = -0.10
    if req_eq.Product.CouponBarrierLevels is not None:
        req_eq.Product.CouponBarrierAbsoluteDigitalSpread = -0.10
    if req_eq.Product.OptionDownAndInBarrierLevel is not None:
        req_eq.Product.OptionDownAndInBarrierAbsoluteDigitalSpread = -0.05
    uids = req_eq.Product.UnderlyingIdentifiers
    # 1) Direct Murray (2016) lower-Cholesky adjoint verification on D=5 equicorrelated (rho=0.65, coalescent eigenvalues).
    corr5 = np.full((5, 5), 0.65, dtype=np.float64)
    np.fill_diagonal(corr5, 1.0)
    chol5 = np.linalg.cholesky(corr5)
    bar_l5 = np.random.default_rng(11).standard_normal((5, 5))
    s5 = cholesky_corr_adjoint_np(chol5, bar_l5)
    murray_sym = s5 + s5.T - np.diag(np.diag(s5))
    h_m = 1e-6
    cu5, cd5 = corr5.copy(), corr5.copy()
    cu5[0, 1] += h_m
    cu5[1, 0] += h_m
    cd5[0, 1] -= h_m
    cd5[1, 0] -= h_m
    fd_m01 = float(np.sum(bar_l5 * (np.linalg.cholesky(cu5) - np.linalg.cholesky(cd5)))) / (2.0 * h_m)
    assert abs(float(murray_sym[0, 1]) - fd_m01) < 1e-9

    # 2) Full QMC CRN run with non-coalescent SPD corr: check device respect, pv_only == all_greeks, and Cega vs FD.
    req_eq.Model.MarketDataSet.Correlations = {
        f"{uids[i]}&{uids[j]}": round(0.45 + 0.04 * i + 0.02 * j, 4)
        for i in range(len(uids))
        for j in range(i + 1, len(uids))
    }
    st_eq = preprocess_eq_autocall(req_eq)
    cfg_ch = QmcAadGreeksConfig(n_paths=2048, n_replicates=4, n_substeps=4, corr_aad_mode="cholesky")

    res_pv = compute_qmc_aad_crn_greeks(st_eq, req_eq, queries=["ProductValue"], config=cfg_ch, device="cpu")
    res_all = compute_qmc_aad_crn_greeks(
        st_eq,
        req_eq,
        queries=["ProductValue", "Delta", "Gamma", "Vega", "Rho", "DividendsSensitivities", "Theta", "Cega"],
        config=cfg_ch,
        device="cpu",
    )
    assert abs(float(res_pv.product_value) - float(res_all.product_value)) < 1e-12
    assert res_all.delta_tensor is not None and res_all.delta_tensor.device.type == "cpu"

    h_c = 2e-3
    r_cu = copy.deepcopy(req_eq)
    r_cd = copy.deepcopy(req_eq)
    k01 = f"{uids[0]}&{uids[1]}"
    c01_base = float(req_eq.Model.MarketDataSet.Correlations[k01])
    r_cu.Model.MarketDataSet.Correlations[k01] = c01_base + h_c
    r_cd.Model.MarketDataSet.Correlations[k01] = c01_base - h_c
    pv_cu = float(
        compute_qmc_aad_crn_greeks(preprocess_eq_autocall(r_cu), r_cu, queries=["ProductValue"], config=cfg_ch, device="cpu").product_value
    )
    pv_cd = float(
        compute_qmc_aad_crn_greeks(preprocess_eq_autocall(r_cd), r_cd, queries=["ProductValue"], config=cfg_ch, device="cpu").product_value
    )
    fd_c01 = (pv_cu - pv_cd) / (2.0 * h_c)
    assert res_all.cega is not None
    got_c01 = float(res_all.cega[uids[0]][uids[1]])
    assert abs(got_c01 - fd_c01) <= max(5e-3, 0.25 * abs(fd_c01))

    # CPU MF-CV vs none convergence.
    cfg_no_cv = QmcAadGreeksConfig(n_paths=2048, n_replicates=4, n_substeps=4, cv_mode="none")
    res_no_cv = compute_qmc_aad_crn_greeks(st_eq, req_eq, queries=["ProductValue"], config=cfg_no_cv, device="cpu")
    assert np.isfinite(float(res_no_cv.product_value))

    # 4-mode ki_obs_type dispatch distinctness.
    pvs_ki: dict[str, float] = {}
    for mode in ("European", "Discrete", "Daily", "Continuous"):
        r_m = copy.deepcopy(req_eq)
        st_m = preprocess_eq_autocall(r_m)
        object.__setattr__(st_m, "ki_obs_type", mode)
        pvs_ki[mode] = float(
            compute_qmc_aad_crn_greeks(st_m, r_m, queries=["ProductValue"], config=cfg_ch, device="cpu").product_value
        )
    assert all(np.isfinite(v) for v in pvs_ki.values())
    assert pvs_ki["Continuous"] <= pvs_ki["European"] + 1e-3


def test_qmc_aad_buehler_lv_rho_divs_and_3d_spotcarry_theta(
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Verifies Buehler LV-rebuild Rho/Dividends pullback and 3d_spotCarry Theta parity."""
    from deepvol.models.autocall.qmc_aad_greeks import (
        QmcAadGreeksConfig,
        _build_lv_table_and_forward_cache,
        _pullback_lv_to_r_nat_and_div,
        compute_qmc_aad_crn_greeks,
    )

    st_base = preprocess_eq_autocall(multi_period_request)
    tk0 = st_base.sorted_identifiers[0]
    surf0 = st_base.vol_surfaces[tk0]
    tau_obs = st_base.tau_obs.detach().cpu().numpy().astype(np.float64)
    r_nat0 = st_base.dividend_schedule.r_native_obs[0].detach().cpu().numpy().astype(np.float64)
    dpv0 = st_base.dividend_schedule.div_pv[0].detach().cpu().numpy().astype(np.float64)
    spot0 = float(st_base.spots_t0[0].item())
    t_grid = np.linspace(1e-5, float(tau_obs[-1]), 32, dtype=np.float64)
    y_grid = np.linspace(-1.5, 1.5, 64, dtype=np.float64)

    cache0 = _build_lv_table_and_forward_cache(surf0, spot0, tau_obs, r_nat0, t_grid, y_grid, dpv0)
    rng = np.random.default_rng(7)
    bar_lv = rng.standard_normal(cache0["lv_out"].shape)
    g_rn, g_dpv = _pullback_lv_to_r_nat_and_div(cache0, bar_lv)

    h = 1e-7
    for m_idx in range(len(tau_obs)):
        rn_p, rn_m = r_nat0.copy(), r_nat0.copy()
        rn_p[m_idx] += h
        rn_m[m_idx] -= h
        lv_p = _build_lv_table_and_forward_cache(surf0, spot0, tau_obs, rn_p, t_grid, y_grid, dpv0)["lv_out"]
        lv_m = _build_lv_table_and_forward_cache(surf0, spot0, tau_obs, rn_m, t_grid, y_grid, dpv0)["lv_out"]
        fd_rn = float(np.sum(bar_lv * (lv_p - lv_m))) / (2.0 * h)
        assert abs(float(g_rn[m_idx]) - fd_rn) <= max(1e-5, 1e-3 * abs(fd_rn))

        dp_p, dp_m = dpv0.copy(), dpv0.copy()
        dp_p[m_idx] += h
        dp_m[m_idx] -= h
        lv_dp = _build_lv_table_and_forward_cache(surf0, spot0, tau_obs, r_nat0, t_grid, y_grid, dp_p)["lv_out"]
        lv_dm = _build_lv_table_and_forward_cache(surf0, spot0, tau_obs, r_nat0, t_grid, y_grid, dp_m)["lv_out"]
        fd_dp = float(np.sum(bar_lv * (lv_dp - lv_dm))) / (2.0 * h)
        assert abs(float(g_dpv[m_idx]) - fd_dp) <= max(1e-5, 1e-3 * abs(fd_dp))

    # Friday 2026-06-05 -> Monday 2026-06-08 3d_spotCarry Theta parity (k_cross==0 vs manual roll & k_cross>0).
    from deepvol.models.autocall.greeks_engine import _curve_df_cached_np, _normalize_curve_obj

    req_fri = copy.deepcopy(multi_period_request)
    req_fri.Model.MarketDataSet.AsOfDate = datetime.datetime(2026, 6, 5, 0, 0, tzinfo=datetime.timezone.utc)
    for curve in req_fri.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = datetime.datetime(2026, 6, 5, 0, 0, tzinfo=datetime.timezone.utc)
    st_fri = preprocess_eq_autocall(req_fri)
    cfg_th = QmcAadGreeksConfig(n_paths=1024, n_replicates=4, n_substeps=4, cv_mode="none")
    res_th = compute_qmc_aad_crn_greeks(
        st_fri,
        req_fri,
        queries=["ProductValue", "Theta", "Rho", "DividendsSensitivities"],
        config=cfg_th,
        theta_convention="murex_bizday_forward_fixed",
        device="cpu",
    )
    assert res_th.theta is not None and np.isfinite(float(res_th.theta))
    assert res_th.rho is not None and len(res_th.rho) >= 1
    assert res_th.dividends_sensitivities is not None

    # Explicit Monday 2026-06-08 manual roll with 3d_spotCarry must match res_th.theta tightly.
    req_mon = copy.deepcopy(req_fri)
    dt_mon = datetime.datetime(2026, 6, 8, 0, 0, tzinfo=datetime.timezone.utc)
    req_mon.Model.MarketDataSet.AsOfDate = dt_mon
    for curve in req_mon.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = dt_mon
    dc_fri = _normalize_curve_obj(st_fri.discount_curve, as_of_date=req_fri.Model.MarketDataSet.AsOfDate)
    df_fin_3d = float(_curve_df_cached_np(dc_fri, [dt_mon])[0])
    for i_tk, tk in enumerate(st_fri.sorted_identifiers):
        df_rep_3d = float(np.exp(-float(st_fri.dividend_schedule.repo_spread_avg[i_tk].item()) * 3.0 / 365.0))
        req_mon.Model.MarketDataSet.Spots[tk] = float(req_mon.Model.MarketDataSet.Spots[tk]) * (df_rep_3d / df_fin_3d)
    st_mon = preprocess_eq_autocall(req_mon)
    res_mon = compute_qmc_aad_crn_greeks(
        st_mon,
        req_mon,
        queries=["ProductValue"],
        config=cfg_th,
        device="cpu",
    )
    manual_theta_3d = float(res_mon.product_value) - float(res_th.product_value)
    assert abs(float(res_th.theta) - manual_theta_3d) < 5e-4

    # k_cross > 0 roll crossing the first future observation date (2026-07-22) with OneStar active.
    from deepvol.api.schemas_eq_autocall import FixingsSpec

    req_cross = copy.deepcopy(req_fri)
    dt_pre = datetime.datetime(2026, 7, 21, 0, 0, tzinfo=datetime.timezone.utc)
    dt_post = datetime.datetime(2026, 7, 22, 0, 0, tzinfo=datetime.timezone.utc)
    req_cross.Model.MarketDataSet.AsOfDate = dt_pre
    for curve in req_cross.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = dt_pre
    req_cross.Product.OneStarStrike = 1.10
    req_cross.Product.OneStarCoupon = 0.0
    for uid in req_cross.Product.UnderlyingIdentifiers:
        spot_v = float(req_cross.Model.MarketDataSet.Spots[uid])
        if uid in req_cross.Model.MarketDataSet.Fixings:
            fx_spec = req_cross.Model.MarketDataSet.Fixings[uid]
            fx_spec.Dates = list(fx_spec.Dates) + [dt_post]
            fx_spec.Values = list(fx_spec.Values) + [spot_v]
        else:
            req_cross.Model.MarketDataSet.Fixings[uid] = FixingsSpec(
                Identifier=uid,
                Dates=[dt_post],
                Values=[spot_v],
            )
    st_cross = preprocess_eq_autocall(req_cross)
    res_cross = compute_qmc_aad_crn_greeks(
        st_cross,
        req_cross,
        queries=["ProductValue", "Theta"],
        config=cfg_th,
        theta_convention="murex_bizday_forward_fixed",
        device="cpu",
    )
    assert res_cross.theta is not None and np.isfinite(float(res_cross.theta))


def test_qmc_aad_is_long_basket_gearing_parity(
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Verifies IsLongBasket with OptionGearing != 1.0 does not double-count gearing in PV or Greeks."""
    import json
    from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig
    from deepvol.models.autocall.dupire_qmc_solver import price_autocall_dupire_qmc
    from deepvol.models.autocall.qmc_aad_greeks import (
        QmcAadGreeksConfig,
        compute_qmc_aad_crn_greeks,
    )

    req_ilb = copy.deepcopy(multi_period_request)
    req_ilb.Product.IsLongBasket = True
    req_ilb.Product.Strike = 0.90
    req_ilb.Product.OptionGearing = 1.0 / 0.90
    req_ilb.Product.OptionDownAndInBarrierLevel = 1.0
    req_ilb.Product.OptionDownAndInBarrierAbsoluteDigitalSpread = -0.10
    req_ilb.Product.AutoCallTriggerAbsoluteDigitalSpread = -0.10
    if req_ilb.Product.CouponBarrierLevels is not None:
        req_ilb.Product.CouponBarrierAbsoluteDigitalSpread = -0.10
    st_ilb = preprocess_eq_autocall(req_ilb)

    cfg_q = QmcAadGreeksConfig(n_paths=2048, n_replicates=4, n_substeps=4)
    res_pv = compute_qmc_aad_crn_greeks(st_ilb, req_ilb, queries=["ProductValue"], config=cfg_q, device="cpu")
    res_all = compute_qmc_aad_crn_greeks(
        st_ilb,
        req_ilb,
        queries=["ProductValue", "Delta", "Gamma", "Vega", "Rho", "Theta", "Cega"],
        config=cfg_q,
        device="cpu",
    )
    assert abs(float(res_pv.product_value) - float(res_all.product_value)) < 1e-12

    res_ref = price_autocall_dupire_qmc(
        st_ilb,
        config=DupireQMCConfig(n_paths=2048, n_replicates=4, n_substeps=4),
    )
    assert abs(float(res_all.product_value) - float(res_ref.product_value)) * 1e4 < 15.0

    # Central FD Delta check on the first underlying confirms consistent scaling with PV.
    uid0 = st_ilb.original_identifiers[0]
    s0 = float(req_ilb.Model.MarketDataSet.Spots[uid0])
    h_s = 0.01 * s0
    ru, rd = copy.deepcopy(req_ilb), copy.deepcopy(req_ilb)
    ru.Model.MarketDataSet.Spots[uid0] = s0 + h_s
    rd.Model.MarketDataSet.Spots[uid0] = s0 - h_s
    pvu = float(compute_qmc_aad_crn_greeks(preprocess_eq_autocall(ru), ru, queries=["ProductValue"], config=cfg_q, device="cpu").product_value)
    pvd = float(compute_qmc_aad_crn_greeks(preprocess_eq_autocall(rd), rd, queries=["ProductValue"], config=cfg_q, device="cpu").product_value)
    fd_delta0 = (pvu - pvd) / (2.0 * h_s)
    assert res_all.delta is not None
    assert abs(float(res_all.delta[uid0]) - fd_delta0) * (0.01 * s0) * 1e4 < 5.0

    # Also verify deal 59553619 directly if the PS batch request fixture is present.
    deal_59553619 = REPO_ROOT / "docs" / "eq_autocall_ps_batch_b10316f2" / "requests" / "trade_095_59553619.json"
    if deal_59553619.is_file():
        req_59 = EQAutocallableTradeRequest.model_validate(json.loads(deal_59553619.read_text()))
        st_59 = preprocess_eq_autocall(req_59)
        cfg_59 = QmcAadGreeksConfig(n_paths=8192, n_replicates=4, n_substeps=8)
        r59_pv = compute_qmc_aad_crn_greeks(st_59, req_59, queries=["ProductValue"], config=cfg_59, device="cpu")
        r59_all = compute_qmc_aad_crn_greeks(
            st_59, req_59, queries=["ProductValue", "CashFlows", "Delta", "Vega"], config=cfg_59, device="cpu"
        )
        r59_ref = price_autocall_dupire_qmc(
            st_59, config=DupireQMCConfig(n_paths=8192, n_replicates=4, n_substeps=8)
        )
        assert abs(float(r59_pv.product_value) - float(r59_all.product_value)) < 1e-12
        assert abs(float(r59_all.product_value) - float(r59_ref.product_value)) * 1e4 < 15.0
        assert abs(float(r59_all.product_value) - 0.8875) * 1e4 < 15.0


@pytest.mark.gpu
def test_gpu_warm_cache_cudagraph_all_greeks_regression(
    multi_period_request: EQAutocallableTradeRequest,
) -> None:
    """Verifies 3 consecutive GPU calls (_warm_gpu_hit=False -> True capture -> True replay) with 0.00 bps drift."""
    import gc

    from deepvol.api.eq_autocall_runtime import ALL_8_GREEK_QUERIES, EQAutocallRuntime
    from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for warm-cache CUDAGraph regression test")

    GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_res = torch.cuda.memory_reserved()

    req_gpu = copy.deepcopy(multi_period_request)
    req_gpu.Queries = ["ProductValue", "CashFlows", *ALL_8_GREEK_QUERIES]  # type: ignore[assignment]
    runtime = EQAutocallRuntime(device="cuda")

    r1 = runtime.price_trade(req_gpu, engine="auto", include_all_greeks=True)
    r2 = runtime.price_trade(req_gpu, engine="auto", include_all_greeks=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r3 = runtime.price_trade(req_gpu, engine="auto", include_all_greeks=True)
    torch.cuda.synchronize()
    call3_ms = (time.perf_counter() - t0) * 1000.0

    for res in (r1, r2, r3):
        assert res.engine_used == "Tier3_DupireQMC"
        assert res.audit_record is not None
        assert res.audit_record.get("greeks_engine_used") == "Tier4_DupireQMC_CRN"
        assert res.Theta is not None and np.isfinite(float(res.Theta))
        assert res.Delta is not None and res.Gamma is not None
        assert res.MultivariateAggregatedCashGamma is not None
        assert res.Vega is not None and res.Rho is not None
        assert res.DividendsSensitivities is not None and res.Cega is not None
        assert len(res.CashFlows) > 0

    assert abs(float(r2.ProductValue) - float(r1.ProductValue)) * 1e4 < 1e-6
    assert abs(float(r3.ProductValue) - float(r1.ProductValue)) * 1e4 < 1e-6
    assert abs(float(r2.Theta) - float(r1.Theta)) <= 1e-8  # type: ignore[arg-type]
    assert abs(float(r3.Theta) - float(r1.Theta)) <= 1e-8  # type: ignore[arg-type]
    assert len(r2.CashFlows) == len(r1.CashFlows) == len(r3.CashFlows)
    for c1, c2, c3 in zip(r1.CashFlows, r2.CashFlows, r3.CashFlows):
        assert abs(float(c2.Amount) - float(c1.Amount)) <= 1e-8
        assert abs(float(c3.Amount) - float(c1.Amount)) <= 1e-8
    for uid in r1.Delta:  # type: ignore[union-attr]
        assert abs(float(r2.Delta[uid]) - float(r1.Delta[uid])) <= 1e-8  # type: ignore[index]
        assert abs(float(r3.Delta[uid]) - float(r1.Delta[uid])) <= 1e-8  # type: ignore[index]

    if sys.gettrace() is None:
        assert call3_ms < 15.0, f"Call 3 warm latency {call3_ms:.2f} ms >= 15.0 ms"

    max_res = torch.cuda.max_memory_reserved()
    if base_res <= 64 * 1024 * 1024:
        assert max_res <= 250.0 * 1024 * 1024, (
            f"Peak reserved VRAM {max_res / (1024.0 * 1024.0):.1f} MB > 250.0 MB"
        )
    else:
        delta_mb = (max_res - base_res) / (1024.0 * 1024.0)
        assert delta_mb <= 250.0, f"Delta reserved VRAM {delta_mb:.1f} MB > 250.0 MB"






