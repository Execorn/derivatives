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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deepvol.api.schemas_eq_autocall import (
    EQAutocallableTradeRequest,
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
REF_REQUEST_PATH = REPO_ROOT / "eq_autocallable_request.txt"


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
    surrogate_f32.zero_init_residual_heads(std=0.0)

    for req in (req_ref, multi_period_request):
        state = preprocess_eq_autocall(req)
        d_assets = state.d_assets
        sorted_ids = state.sorted_identifiers

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
            assert res.delta_tensor is not None
            assert res.rho_arrays is not None
            assert res.dividends_arrays is not None
            assert res.vega_arrays is not None
            assert res.cega_tensor is not None

            # 1. Delta vs Central Finite Differences on spots_t0
            eps_s = 1e-5
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
            h_rate = 1e-4  # 0.01 bps in percentage quote units
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
            h_div = 1e-4
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

                if np.linalg.norm(fd_div) > 1e-6:
                    rel_div = _rel_err(got_div, fd_div)
                    assert rel_div <= 1e-5, f"DividendsSensitivities[{ticker}] rel err {rel_div:.3e} > 1e-5"
                    if surr is not surrogate_f32:
                        assert rel_div <= 1e-7
                else:
                    assert float(np.linalg.norm(got_div - fd_div)) <= 1e-11

            # 4. Vega & Cega vs Central Finite Differences (spot-checked on active asset/pair)
            if surr is not surrogate_f32:
                active_vega_ticker = max(
                    sorted_ids, key=lambda t: float(np.max(np.abs(res.vega_arrays[t])))
                )
                got_vega0 = res.vega_arrays[active_vega_ticker]
                e_idx, k_idx = np.unravel_index(int(np.argmax(np.abs(got_vega0))), got_vega0.shape)
                h_vol = 1e-6
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

    # Warmup
    _ = compute_batched_surrogate_greeks(
        surrogate, node_b, edge_b, amask_b, sched_b, smask_b, glob_b, compute_cash_gamma=True
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

    if device.type == "cuda":
        assert ms_per_trade <= 0.05, (
            f"Gate G2.4 violation: batched GPU Greeks took {ms_per_trade:.4f} ms/trade > 0.05 ms/trade"
        )
