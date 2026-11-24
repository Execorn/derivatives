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
                    assert _rel_err(res.vega_arrays[tk_ref], v_ref) <= 1e-5
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
            h_div = 1e-1 if surr is surrogate_f32 else 5e-4
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
                    tol_small = 2e-6 if surr is surrogate_f32 else 1e-8
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

            # Verify against second-order central finite difference on S_0 * (1 + eps)
            h_eps = 1e-4
            base_s = state.spots_t0.detach()
            pv_0 = float(evaluate_differentiable_pv(state, spots_t0=base_s, surrogate=surr).item())
            pv_p = float(
                evaluate_differentiable_pv(state, spots_t0=base_s * (1.0 + h_eps), surrogate=surr).item()
            )
            pv_m = float(
                evaluate_differentiable_pv(state, spots_t0=base_s * (1.0 - h_eps), surrogate=surr).item()
            )
            fd_cash_gamma = (pv_p - 2.0 * pv_0 + pv_m) / (h_eps * h_eps)
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



