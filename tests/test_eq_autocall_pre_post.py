"""test_eq_autocall_pre_post.py — Verification suite for Task 1.1 & Task 1.3 (Tier-1 Pre/Post-Processing).

Covers:
  - Gate G1.3: Exact 37-row CashFlows reconciliation on `eq_autocallable_request.txt`
    (Settled Periods 1-10, In-Transit Period 11 with StandardError=0.0, Future Period 12,
    memory coupon accumulator c_mem_0 = 0.32816, DF_12 = 0.9594909887002673).
  - Gate G4.4: Strict float64 / float32 precision segregation audit across all M1 modules.
  - Lexicographical asset canonicalization pi_lex and inverse restoration pi_lex^{-1}.
  - Sentinel barrier normalization (KO >= 10.0 -> non-callable, Cpn <= 1e-4 -> guaranteed).
  - Escrowed dividend forward projection, PredefinedDividends override, and missing RepoSpreadCurves.
  - Differentiable autograd graph connectivity with respect to spot prices S_i(t_0).
  - W1 affine payoff modifiers (NoNotionalRepayment, AutoCallCoupon, ForceCouponsOnAutocall, FundingLeg).
"""

from __future__ import annotations

import ast
import copy
import datetime
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
from deepvol.models.autocall import (
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    assemble_autocall_cashflows_and_pv,
    normalize_sentinel_barriers,
    preprocess_eq_autocall,
    price_eq_autocall_dupire_qmc,
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


def test_G1_3_reference_trade_37_cashflows_reconciliation(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G1.3: Exact 37-row CashFlows state machine, memory accumulator, and In-Transit freeze."""
    req_model, expected_res = parsed_dag
    expected_cfs = expected_res["CashFlows"]
    assert len(expected_cfs) == 37

    state_from_path = preprocess_eq_autocall(REF_REQUEST_PATH)
    state_from_model = preprocess_eq_autocall(req_model)

    # Verify memory coupon accumulation from Periods 4..10 (7 missed periods * 0.04688 = 0.32816)
    assert state_from_path.c_mem_0 == pytest.approx(0.32816, abs=1e-12)
    assert state_from_model.c_mem_0 == pytest.approx(0.32816, abs=1e-12)
    assert state_from_path.m_rem == 1
    assert state_from_path.pv_in_transit == pytest.approx(0.0, abs=1e-15)

    # Verify historical period partitioning: 10 Settled + 1 In-Transit = 11 historical records
    hist = state_from_path.seasoned_state.historical_records
    assert len(hist) == 11
    for p_idx in range(10):
        assert hist[p_idx].status == "Settled"
    assert hist[10].period_index == 10
    assert hist[10].status == "In-Transit"
    assert hist[10].coupon_paid == pytest.approx(0.0, abs=1e-15)
    assert hist[10].notional_paid == pytest.approx(0.0, abs=1e-15)

    # Verify Period 12 discount factor DF(t_0, T_12 = 2027-02-09)
    assert float(state_from_path.df_pay[0].item()) == pytest.approx(
        0.9594909887002673, abs=1e-12
    )

    # Reconstruct all 37 CashFlows rows using the exact reference terminal put loss
    ref_opt_pv = float(expected_res["Option"])
    df_12 = float(state_from_path.df_pay[0].item())
    gearing = state_from_path.option_gearing
    l_put_ref = torch.tensor(-ref_opt_pv / (df_12 * gearing), dtype=torch.float64)

    post = assemble_autocall_cashflows_and_pv(
        state_from_model,
        p_stop=torch.zeros(1, dtype=torch.float64),
        q_cpn=torch.zeros(1, dtype=torch.float64),
        l_put=l_put_ref,
        option_se=float(expected_res["StandardError"]),
    )

    assert len(post.cashflows) == 37
    assert float(post.notional_pv.item()) == pytest.approx(0.9594909887002673, abs=1e-12)
    assert float(post.coupon_pv.item()) == pytest.approx(0.0, abs=1e-12)
    assert float(post.option_pv.item()) == pytest.approx(ref_opt_pv, abs=1e-12)
    assert float(post.product_value.item()) == pytest.approx(
        float(expected_res["ProductValue"]), abs=1e-12
    )

    for idx, (cf_got, cf_exp) in enumerate(zip(post.cashflows, expected_cfs)):
        assert cf_got.Description == cf_exp["Description"], f"Row {idx} Description mismatch"
        exp_dt = cf_exp["Date"]
        if isinstance(exp_dt, datetime.datetime):
            assert cf_got.Date.date() == exp_dt.date(), f"Row {idx} Date mismatch"
        assert cf_got.StatusText == cf_exp["StatusText"], f"Row {idx} StatusText mismatch"
        assert cf_got.Amount == pytest.approx(float(cf_exp["Amount"]), abs=1e-12), (
            f"Row {idx} ({cf_got.Description}) Amount mismatch"
        )
        assert cf_got.PaymentValue == pytest.approx(float(cf_exp["Amount"]), abs=1e-12)
        if idx < 33:
            # Rows 0..29 (Settled Periods 1..10) and Rows 30..32 (In-Transit Period 11)
            assert cf_got.StandardError == 0.0, f"Row {idx} must have zero StandardError"
        else:
            assert cf_got.StandardError == pytest.approx(
                float(cf_exp["StandardError"]), abs=1e-12
            )


def test_G4_4_precision_segregation_audit(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G4.4: Verify strict float64 execution in Tier-1/Tier-3 and float32 cast only at Tier-2 boundary."""
    req_model, _ = parsed_dag
    m1_dir = REPO_ROOT / "src" / "deepvol" / "models" / "autocall"
    strict_f64_files = [
        m1_dir / "seasoned_collapser.py",
        m1_dir / "escrowed_dividend.py",
        m1_dir / "analytical_baseline.py",
        m1_dir / "dupire_qmc_solver.py",
    ]

    # 1. Static AST audit: zero float32/float16/bfloat16 references in pure Tier-1/Tier-3 modules
    forbidden_attrs = {"float32", "float16", "bfloat16", "half"}
    for path in strict_f64_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                pytest.fail(f"Forbidden precision attribute '{node.attr}' found in {path.name}:{node.lineno}")

    # In pre_post_processor.py, float32 is permitted ONLY inside `to_tier2_float32_tensors`
    pre_post_path = m1_dir / "pre_post_processor.py"
    pp_tree = ast.parse(pre_post_path.read_text(encoding="utf-8"), filename=str(pre_post_path))
    for node in ast.walk(pp_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "to_tier2_float32_tensors":
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and child.attr in forbidden_attrs:
                    pytest.fail(
                        f"Forbidden precision attribute '{child.attr}' outside to_tier2_float32_tensors "
                        f"in {pre_post_path.name}:{child.lineno}"
                    )

    # 2. Runtime tensor dtype audit across the full M1 pipeline
    state = preprocess_eq_autocall(req_model)
    for field_name, val in vars(state).items():
        if isinstance(val, torch.Tensor):
            assert val.dtype == torch.float64, f"AutocallPreprocState.{field_name} has dtype {val.dtype}"

    for field_name, val in vars(state.dividend_schedule).items():
        if isinstance(val, torch.Tensor):
            assert val.dtype == torch.float64, f"EscrowedDividendOutput.{field_name} has dtype {val.dtype}"

    for field_name, val in vars(state.baseline).items():
        if isinstance(val, torch.Tensor):
            assert val.dtype == torch.float64, f"AutocallBaselineOutput.{field_name} has dtype {val.dtype}"

    qmc_res = price_eq_autocall_dupire_qmc(req_model, n_paths=2048)
    for field_name, val in vars(qmc_res).items():
        if isinstance(val, torch.Tensor):
            assert val.dtype == torch.float64, f"DupireQMCResult.{field_name} has dtype {val.dtype}"
        elif isinstance(val, np.ndarray):
            assert val.dtype == np.float64, f"DupireQMCResult.{field_name} has dtype {val.dtype}"

    # Verify Tier-2 boundary cast produces float32 tensors with expected dimensions
    tier2 = state.to_tier2_float32_tensors()
    assert set(tier2.keys()) == {
        "node_feats",
        "edge_feats",
        "asset_mask",
        "sched_feats",
        "sched_mask",
        "global_feats",
    }
    for k, t_val in tier2.items():
        assert t_val.dtype == torch.float32, f"to_tier2_float32_tensors[{k!r}] has dtype {t_val.dtype}"
    assert tier2["node_feats"].shape == (1, 8, len(AutocallNodeCol))
    assert tier2["edge_feats"].shape == (1, 8, 8, len(AutocallEdgeCol))
    assert tier2["sched_feats"].shape == (1, 48, len(AutocallSchedCol))
    assert tier2["global_feats"].shape == (1, len(AutocallGlobalCol))


def test_canonical_asset_lex_permutation_and_inverse(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify permutation invariance under pi_lex and exact sensitivity restoration under pi_lex^{-1}."""
    req_orig, _ = parsed_dag
    req_perm = copy.deepcopy(req_orig)

    # Permute the 5 UnderlyingIdentifiers and UnderlyingReferenceSpots
    perm_idx = [3, 1, 4, 0, 2]
    orig_ids = list(req_orig.Product.UnderlyingIdentifiers)
    orig_refs = list(req_orig.Product.UnderlyingReferenceSpots)
    req_perm.Product.UnderlyingIdentifiers = [orig_ids[i] for i in perm_idx]
    req_perm.Product.UnderlyingReferenceSpots = [orig_refs[i] for i in perm_idx]

    s_orig = preprocess_eq_autocall(req_orig)
    s_perm = preprocess_eq_autocall(req_perm)

    expected_canonical = [
        "MAGN_RUB",
        "POSIRX_RUB",
        "SIBN_RUB",
        "TATN_RUB",
        "VKCO_RUB",
    ]
    assert s_orig.sorted_identifiers == expected_canonical
    assert s_perm.sorted_identifiers == expected_canonical

    torch.testing.assert_close(s_orig.node_feats, s_perm.node_feats, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(s_orig.edge_feats, s_perm.edge_feats, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(s_orig.sched_feats, s_perm.sched_feats, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(s_orig.global_feats, s_perm.global_feats, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        s_orig.baseline.product_value_proxy,
        s_perm.baseline.product_value_proxy,
        rtol=1e-13,
        atol=1e-13,
    )

    # Verify inverse permutation restores caller's asset ordering
    canon_vec = torch.arange(1, 6, dtype=torch.float64)
    restored_orig = s_orig.unpermute_assets(canon_vec)
    restored_perm = s_perm.unpermute_assets(canon_vec)
    for idx_in_perm, orig_idx in enumerate(perm_idx):
        assert float(restored_perm[idx_in_perm].item()) == float(restored_orig[orig_idx].item())


def test_sentinel_barrier_normalization() -> None:
    """Verify sentinel barrier normalization maps KO >= 10.0 to non-callable and Cpn <= 1e-4 to guaranteed."""
    res = normalize_sentinel_barriers(
        ko_barriers=[500.0, 0.97, 500.0],
        cpn_barriers=[1e-5, 0.75, 0.75],
    )
    assert list(res.is_callable) == [0.0, 1.0, 0.0]
    assert list(res.log_ko_barrier_feat) == pytest.approx(
        [
            float(np.log(3.0)),
            float(np.log(0.97)),
            float(np.log(3.0)),
        ]
    )
    assert list(res.is_guaranteed_cpn) == [1.0, 0.0, 0.0]
    assert list(res.log_cpn_barrier_feat) == pytest.approx(
        [
            float(np.log(0.05)),
            float(np.log(0.75)),
            float(np.log(0.75)),
        ]
    )


def test_escrowed_dividend_priority_and_missing_repo(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify PredefinedDividends override MarketData.Dividends and missing RepoSpreadCurves default to 0.0."""
    req_orig, _ = parsed_dag
    state_base = preprocess_eq_autocall(req_orig)

    # POSIRX_RUB has no RepoSpreadCurves entry -> repo_spread_avg must be 0.0 and df_repo_obs == 1.0
    posirx_idx = state_base.sorted_identifiers.index("POSIRX_RUB")
    assert float(state_base.dividend_schedule.repo_spread_avg[posirx_idx].item()) == 0.0
    assert float(state_base.dividend_schedule.df_repo_obs[posirx_idx, 0].item()) == pytest.approx(1.0, abs=1e-15)

    # Override VKCO_RUB (which has 0 market dividends) via PredefinedDividends with a discrete cash dividend
    vkco_idx = state_base.sorted_identifiers.index("VKCO_RUB")
    assert float(state_base.dividend_schedule.div_pv[vkco_idx, 0].item()) == 0.0

    req_override = copy.deepcopy(req_orig)
    req_override.Product.PredefinedDividends = {
        "VKCO_RUB": PredefinedDividendsSpec(
            ExDates=[datetime.datetime(2026, 12, 15, 0, 0, tzinfo=datetime.timezone.utc)],
            Amounts=[25.0],
        )
    }
    state_override = preprocess_eq_autocall(req_override)
    assert float(state_override.dividend_schedule.div_pv[vkco_idx, 0].item()) > 20.0
    assert float(state_override.forward_perf[vkco_idx, 0].item()) < float(
        state_base.forward_perf[vkco_idx, 0].item()
    )


def test_autograd_differentiability_through_preprocessor(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify exact PyTorch float64 autograd connectivity from spots_t0 through forward_perf and baseline PV."""
    req_model, _ = parsed_dag
    s0 = preprocess_eq_autocall(req_model)
    d_assets = s0.d_assets
    spots_var = s0.spots_t0.detach().clone().requires_grad_(True)

    s_grad = preprocess_eq_autocall(req_model, spots_override=spots_var)
    pv_proxy = s_grad.baseline.product_value_proxy
    assert pv_proxy.requires_grad

    (grad_auto,) = torch.autograd.grad(pv_proxy, spots_var, create_graph=False)
    assert grad_auto.dtype == torch.float64

    # Central finite difference check on all active canonical assets
    eps = 1e-5
    grad_fd = torch.zeros_like(grad_auto)
    for i in range(d_assets):
        sp_up = s0.spots_t0.detach().clone()
        sp_dn = s0.spots_t0.detach().clone()
        h = float(sp_up[i].item()) * eps
        sp_up[i] += h
        sp_dn[i] -= h
        pv_up = preprocess_eq_autocall(req_model, spots_override=sp_up).baseline.product_value_proxy.item()
        pv_dn = preprocess_eq_autocall(req_model, spots_override=sp_dn).baseline.product_value_proxy.item()
        grad_fd[i] = (pv_up - pv_dn) / (2.0 * h)

    torch.testing.assert_close(grad_auto, grad_fd, rtol=1e-5, atol=1e-7)


def test_w1_payoff_modifiers_assembly(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify NoNotionalRepayment, AutoCallCoupon, ForceCouponsOnAutocall, and FundingLeg assembly."""
    req_orig, _ = parsed_dag
    req_mod = copy.deepcopy(req_orig)
    req_mod.Product.NoNotionalRepayment = True
    req_mod.Product.AutoCallCoupon = 0.05
    req_mod.Product.ForceCouponsOnAutocall = True
    req_mod.Product.AllowAutocallOnExpiry = True
    req_mod.Product.AutoCallTriggerLevels[-1] = 0.90
    req_mod.Product.RateIdentifier = "RUB_KEY_RATE"
    req_mod.Product.FundingSpread = 0.03
    req_mod.Product.FundingLegPaymentDates = [
        datetime.datetime(2027, 2, 9, 0, 0, tzinfo=datetime.timezone.utc)
    ]

    state = preprocess_eq_autocall(req_mod)
    assert state.no_notional_repayment is True
    assert state.autocall_coupon == pytest.approx(0.05, abs=1e-15)
    assert state.force_coupons_on_autocall is True
    assert state.funding_leg is not None

    post = assemble_autocall_cashflows_and_pv(
        state,
        p_stop=torch.tensor([0.40], dtype=torch.float64),
        q_cpn=torch.tensor([0.10], dtype=torch.float64),
        l_put=torch.tensor(0.20, dtype=torch.float64),
    )
    # Since NoNotionalRepayment is True, future notional repayment is 0.0
    assert float(post.notional_pv.item()) == pytest.approx(0.0, abs=1e-15)
    assert float(post.funding_pv.item()) > 0.0
    # Since ForceCouponsOnAutocall is True, effective coupon trigger is max(q_m, p_m / S_{m-1}) = 0.40
    df_12 = float(state.df_pay[0].item())
    eff_cpn = float(state.coupon_amounts[0].item()) + state.c_mem_0
    expected_cpn_pv = df_12 * (0.40 * eff_cpn + 0.40 * 0.05)
    assert float(post.coupon_pv.item()) == pytest.approx(expected_cpn_pv, abs=1e-12)


def test_m1_in_transit_expiry_and_cash_to_settle_today(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    req_orig, _ = parsed_dag

    # Case 1: SC-017/018/019 In-Transit final maturity (t_11 <= AsOfDate < T_11), ITM put + LowerStrike + OneStar + PartCall
    req_it = copy.deepcopy(req_orig)
    final_obs = req_it.Product.ObservationDates[-1]
    final_pay = req_it.Product.PaymentDates[-1]
    assert final_obs < final_pay
    as_of_it = final_obs + datetime.timedelta(days=1)
    req_it.Model.MarketDataSet.AsOfDate = as_of_it
    for curve in req_it.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = as_of_it
    # Add fixing on final_obs with ITM put (50% of ref spot)
    for uid, ref_s in zip(req_it.Product.UnderlyingIdentifiers, req_it.Product.UnderlyingReferenceSpots):
        fix = req_it.Model.MarketDataSet.Fixings[uid]
        fix.Dates.append(final_obs)
        fix.Values.append(0.50 * float(ref_s))

    req_it.Product.LowerStrike = 0.60
    req_it.Product.ParticipationCallStrike = 0.40
    req_it.Product.ParticipationCallRate = 0.25
    req_it.Product.OneStarStrike = 1.20
    req_it.Product.OneStarCoupon = 0.03
    req_it.Product.OptionDownAndInBarrierLevel = 0.70

    st_it = preprocess_eq_autocall(req_it)
    assert st_it.m_rem == 0
    assert st_it.seasoned_state.already_autocalled is False
    df_last = float(st_it.discount_curve.df(final_pay))
    # Put capped at Strike - LowerStrike = 0.90 - 0.60 = 0.30; Opt = -gearing * 0.30 + 0.25 * (0.50 - 0.40)
    expected_opt = -float(req_it.Product.OptionGearing) * 0.30 + 0.25 * 0.10
    expected_pv = df_last * (1.0 + expected_opt)
    assert st_it.pv_in_transit == pytest.approx(expected_pv, abs=1e-11)

    asm = assemble_autocall_cashflows_and_pv(
        st_it,
        p_stop=torch.zeros(0, dtype=torch.float64),
        q_cpn=torch.zeros(0, dtype=torch.float64),
        l_put=torch.tensor(0.0, dtype=torch.float64),
    )
    assert float(asm.product_value.item()) == pytest.approx(expected_pv, abs=1e-11)
    assert any(cf.Description == "Option" and cf.StatusText == "Estimate" for cf in asm.cashflows)

    # Case 2: EXT-016 CashToSettleToday when AsOfDate == final PayDate
    req_settle = copy.deepcopy(req_it)
    req_settle.Model.MarketDataSet.AsOfDate = final_pay
    for curve in req_settle.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = final_pay
    st_settle = preprocess_eq_autocall(req_settle)
    assert st_settle.seasoned_state.cash_to_settle_today == pytest.approx(1.0 + expected_opt, abs=1e-11)
    assert any(cf.StatusText == "SettlesToday" for cf in st_settle.seasoned_state.past_cashflows)
    assert all(
        cf.StatusText == ("SettlesToday" if cf.Date.date() == final_pay.date() else "Settled")
        for cf in st_settle.seasoned_state.past_cashflows
    )
    asm_settle = assemble_autocall_cashflows_and_pv(
        st_settle,
        p_stop=torch.zeros(0, dtype=torch.float64),
        q_cpn=torch.zeros(0, dtype=torch.float64),
        l_put=torch.tensor(0.0, dtype=torch.float64),
    )
    assert asm_settle.cash_to_settle_today == pytest.approx(1.0 + expected_opt, abs=1e-11)
    assert float(asm_settle.product_value.item()) == pytest.approx(1.0 + expected_opt, abs=1e-11)


def test_m1_quanto_dividends_and_fx_tenors(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    req_orig, _ = parsed_dag
    req_q = copy.deepcopy(req_orig)
    as_of = req_q.Model.MarketDataSet.AsOfDate
    # 2-asset basket: USD asset vs RUB asset paid in RUB, unsorted to also test SCH-004 role bits
    req_q.Product.UnderlyingIdentifiers = ["VKCO_RUB", "AAPL_USD"]
    req_q.Product.UnderlyingReferenceSpots = [100.0, 100.0]
    req_q.Product.CallabilityMonitoringIndex = "RatioAB"
    req_q.Product.CouponBarrierMonitoringIndex = "DiffAB"
    req_q.Product.OptionMonitoringIndex = "RatioBA"
    req_q.Product.FixedCouponAmount = [0.02] * 12
    req_q.Product.PredefinedDividends = {
        "AAPL_USD": PredefinedDividendsSpec(
            ExDates=[as_of + datetime.timedelta(days=45)],
            Amounts=[10.0],
        )
    }
    mds = req_q.Model.MarketDataSet
    mds.Spots = {"VKCO_RUB": 100.0, "AAPL_USD": 100.0}
    mds.Fixings["AAPL_USD"] = {
        "Identifier": "AAPL_USD",
        "Dates": list(mds.Fixings["VKCO_RUB"].Dates),
        "Values": [90.0] * len(mds.Fixings["VKCO_RUB"].Dates),
    }
    mds.Fixings["VKCO_RUB"].Values = [85.0] * len(mds.Fixings["VKCO_RUB"].Dates)
    mds.Correlations = {"VKCO_RUB&AAPL_USD": 0.40, "AAPL_USD&USD/RUB": 0.60, "VKCO_RUB&USD/RUB": 0.10}
    mds.RatesCurvesBundles["USD_SOFR"] = copy.deepcopy(mds.RatesCurvesBundles["RUB_RUONIA_OIS"])
    mds.RatesCurvesBundles["USD_SOFR"].Identifier = "USD_SOFR"
    mds.RatesCurvesBundles["USD_SOFR"].Currency = "USD"
    mds.RatesCurvesBundles["USD_SOFR"].RatesSchedule.Rates = [4.0] * len(
        mds.RatesCurvesBundles["USD_SOFR"].RatesSchedule.Dates
    )
    req_q.Model.FinancingCurveIdentifiers = ["RUB_RUONIA_OIS", "USD_SOFR"]
    mds.VolatilitySurfaces["AAPL_USD"] = copy.deepcopy(mds.VolatilitySurfaces["VKCO_RUB"])
    mds.VolatilitySurfaces["AAPL_USD"].Identifier = "AAPL_USD"
    mds.VolatilitySurfaces["AAPL_USD"].Strikes = [50.0, 80.0, 100.0, 120.0, 150.0]
    n_exp = len(mds.VolatilitySurfaces["AAPL_USD"].ExpirationDates)
    mds.VolatilitySurfaces["AAPL_USD"].VolatilityQuotes = [[0.25] * 5 for _ in range(n_exp)]
    mds.VolatilitySurfaces["USD/RUB"] = {
        "Identifier": "USD/RUB",
        "TypeName": "FXVolatilitySmileCollection",
        "DayCountConvention": "Act365",
        "SmileInterpolationMethod": "Smooth",
        "SmileExtrapolationMethod": "Linear",
        "Deltas": ["-25D", "ATM", "25D"],
        "Tenors": ["O/N", "TN", "SN", "0.5Y", "1.5Y"],
        "VolatilityQuotes": [[0.20, 0.20, 0.20] for _ in range(5)],
    }

    st = preprocess_eq_autocall(req_q)
    # Lex sorted order is ['AAPL_USD', 'VKCO_RUB'] while caller order was ['VKCO_RUB', 'AAPL_USD']
    assert st.sorted_identifiers == ["AAPL_USD", "VKCO_RUB"]
    assert tuple(st.inv_lex_perm.tolist()) == (1, 0)
    assert float(st.node_feats[0, 0, AutocallNodeCol.ROLE_LEG_B].item()) == 1.0
    assert float(st.node_feats[0, 1, AutocallNodeCol.ROLE_LEG_A].item()) == 1.0
    assert "AAPL_USD" in st.dividend_schedule.quanto_meta
    # Exact Q^pay quanto forward check: F = (S0 - D_pv) * exp((r_f - q_repo - rho_s_fx*sig_s*sig_fx)*tau)
    aapl_fwd = float(st.dividend_schedule.forward_price[0, 0].item())
    assert 80.0 < aapl_fwd < 100.0

    # T1M-018: smooth softplus floor when D_pv >= 0.80 * S_0
    req_huge_div = copy.deepcopy(req_q)
    req_huge_div.Product.PredefinedDividends["AAPL_USD"] = PredefinedDividendsSpec(
        ExDates=[as_of + datetime.timedelta(days=45)],
        Amounts=[95.0],
    )
    st_hd = preprocess_eq_autocall(req_huge_div)
    assert float(st_hd.dividend_schedule.escrowed_spot[0].item()) > 0.0
    assert torch.isfinite(st_hd.forward_perf).all()


def test_m1_seasoned_out_down_ki_history_list_coupons_and_funding_fixings(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    req_orig, _ = parsed_dag
    req = copy.deepcopy(req_orig)
    # List FixedRate + CouponBarrierType='Out' + CouponBarrierDirection='Down' + KI past breach + FloatingFundingLeg
    req.Product.FixedCouponAmount = None
    req.Product.FixedRate = [0.08 + 0.01 * m for m in range(12)]
    req.Product.CouponBarrierType = "Out"
    req.Product.CouponBarrierDirection = "Down"
    req.Product.OptionDownAndInBarrierLevel = 0.70
    req.Product.OptionDownAndInBarrierObservationType = "Discrete"
    req.Product.RateIdentifier = "RUB_KEY_RATE"
    req.Product.FundingSpread = 0.015
    req.Product.FundingLegPaymentDates = list(req.Product.PaymentDates)
    req.Model.MarketDataSet.Fixings["RUB_KEY_RATE"] = {
        "Identifier": "RUB_KEY_RATE",
        "Dates": [d.isoformat() for d in req.Product.ObservationDates[:11]],
        "Values": [0.16] * 11,
    }
    st = preprocess_eq_autocall(req)
    assert st.seasoned_state.i_ki_past == 1.0
    assert st.coupon_barrier_type == "Out"
    assert st.coupon_barrier_direction == "Down"
    assert st.funding_leg is not None
    assert len(st.seasoned_state.past_cashflows) > 0

    # Warrant funding (no RateIdentifier, FundingSpread only)
    req_war = copy.deepcopy(req)
    req_war.Product.RateIdentifier = None
    req_war.Product.FundingSpread = 0.02
    st_war = preprocess_eq_autocall(req_war)
    assert st_war.funding_leg is not None


def test_m1_cache_mutation_immunity_and_import_hygiene(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    import os
    import deepvol
    from deepvol.models.autocall.feature_layout import monitoring_index_to_code

    assert "NUMBA_DISABLE_JIT" not in os.environ
    assert monitoring_index_to_code(None) == 0
    assert deepvol.__version__ is not None

    req_orig, _ = parsed_dag
    req_base = copy.deepcopy(req_orig)
    req_base.Model.MarketDataSet.AsOfDate = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    for c in req_base.Model.MarketDataSet.RatesCurvesBundles.values():
        c.AsOfDate = req_base.Model.MarketDataSet.AsOfDate

    st0 = preprocess_eq_autocall(req_base)
    pv0 = float(st0.baseline.product_value_proxy.item())

    # 1. Interior vol quote mutation (corners [0][0] and [-1][-1] unchanged) -> MUST NOT collide
    req_vol = copy.deepcopy(req_base)
    vq = req_vol.Model.MarketDataSet.VolatilitySurfaces["MAGN_RUB"].VolatilityQuotes
    vq[1] = [float(x) + 0.08 for x in vq[1]]
    pv_vol = float(preprocess_eq_autocall(req_vol).baseline.product_value_proxy.item())
    assert abs(pv_vol - pv0) > 1e-6

    # 2. Repo rate mutation -> MUST NOT collide
    req_repo = copy.deepcopy(req_base)
    req_repo.Model.MarketDataSet.RepoSpreadCurves["VKCO_RUB"].RatesSchedule.Rates = [25.0] * len(
        req_repo.Model.MarketDataSet.RepoSpreadCurves["VKCO_RUB"].RatesSchedule.Rates
    )
    pv_repo = float(preprocess_eq_autocall(req_repo).baseline.product_value_proxy.item())
    assert abs(pv_repo - pv0) > 1e-6

    # 3. Historical fixing mutation flipping to already_autocalled -> MUST NOT collide
    req_fix = copy.deepcopy(req_base)
    for uid, ref_s in zip(req_fix.Product.UnderlyingIdentifiers, req_fix.Product.UnderlyingReferenceSpots):
        req_fix.Model.MarketDataSet.Fixings[uid].Values[3] = float(ref_s) * 1.50
    st_fix = preprocess_eq_autocall(req_fix)
    assert st_fix.seasoned_state.already_autocalled is True

    # 4. Mutating returned AutocallPreprocState does not poison subsequent cache hits
    st_clone_a = preprocess_eq_autocall(req_base)
    st_clone_a.spots_t0.fill_(1.0)
    st_clone_b = preprocess_eq_autocall(req_base)
    assert float(st_clone_b.spots_t0[0].item()) > 10.0

    # 5. Raw dict Spots <= 0 must raise ValueError defensively (EXT-003)
    raw_bad = req_base.model_dump()
    raw_bad.pop("stripped_legacy_fields", None)
    raw_bad["Model"]["MarketDataSet"]["Spots"]["MAGN_RUB"] = 0.0
    with pytest.raises(ValueError):
        preprocess_eq_autocall(raw_bad)


def test_m1_escrowed_dividend_helpers_lazy_jacobian_and_state_aliases(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    from deepvol.market.eq_curves import ClampedYieldCurve, RepoCurve
    from deepvol.models.autocall import escrowed_dividend as ed

    as_of = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    obs_dates = [as_of + datetime.timedelta(days=90), as_of + datetime.timedelta(days=180)]

    assert ed._parse_period_days("2W") == pytest.approx(14.0)
    assert ed._parse_period_days("3M") == pytest.approx(90.0)
    assert ed._parse_period_days("1Y") == pytest.approx(365.0)
    assert ed._parse_period_days("5D") == pytest.approx(5.0)
    assert ed._parse_period_days(30.0) == pytest.approx(30.0)
    assert ed._parse_period_days("O/N") == pytest.approx(1.0)
    assert ed._parse_period_days("T/N") == pytest.approx(2.0)
    assert ed._parse_period_days("S/N") == pytest.approx(3.0)
    assert ed._parse_tenor_years(365.0, "Act365") == pytest.approx(1.0)
    assert ed._parse_tenor_years("2W", "Act365") == pytest.approx(2.0 / 52.0)
    assert ed._parse_tenor_years("6M", "Act365") == pytest.approx(0.5)
    assert ed._parse_tenor_years("1Y", "Bus252") == pytest.approx(1.0)
    assert ed._parse_tenor_years("5D", "Bus252") == pytest.approx(5.0 / 252.0)
    assert ed._parse_tenor_years("30D", "Act365") == pytest.approx(30.0 / 365.0)
    with pytest.raises(ValueError):
        ed._parse_period_days("invalid")

    curve_obj = ClampedYieldCurve(as_of_date=as_of, pillar_tenors=[0.5, 1.0], rates=[12.0, 12.0])
    repo_obj = RepoCurve(as_of_date=as_of, periods_days=[180.0, 365.0], rates=[1.0, 1.0])
    assert ed.build_repo_rate_evaluator(repo_obj, as_of)(obs_dates[0]) == pytest.approx(0.01)
    assert ed.build_repo_rate_evaluator(curve_obj, as_of)(obs_dates[0]) == pytest.approx(0.12)
    assert ed.build_repo_rate_evaluator(None, as_of)(obs_dates[0]) == pytest.approx(0.0)
    repo_dict = {"AsOfDate": as_of.isoformat(), "RatesSchedule": {"Periods": ["180D", "365D"], "Rates": [2.0, 2.0]}}
    assert ed.build_repo_rate_evaluator(repo_dict, as_of)(obs_dates[0]) == pytest.approx(0.02)

    r_none, df_none = ed.build_discount_rate_evaluator(None, as_of)
    assert r_none(obs_dates[0]) == pytest.approx(0.0)
    assert df_none(obs_dates[0]) == pytest.approx(1.0)
    disc_dict = {
        "AsOfDate": as_of.isoformat(),
        "RatesSchedule": {
            "Dates": [(as_of + datetime.timedelta(days=180)).isoformat(), (as_of + datetime.timedelta(days=365)).isoformat()],
            "Rates": [10.0, 10.0],
        },
    }
    r_dict, df_dict = ed.build_discount_rate_evaluator(disc_dict, as_of)
    assert r_dict(obs_dates[0]) == pytest.approx(0.10)
    assert df_dict(obs_dates[0]) < 1.0

    class _DfOnlyObj:
        def df(self, dt: Any) -> float:
            return 0.95

    r_df_only, df_df_only = ed.build_discount_rate_evaluator(_DfOnlyObj(), as_of)
    assert df_df_only(obs_dates[0]) == pytest.approx(0.95)
    assert r_df_only(obs_dates[0]) > 0.0

    ex_d = as_of + datetime.timedelta(days=30)
    evs_dict_ex = ed._extract_events_from_div_source({"ExDates": [ex_d], "Amounts": [2.5]})
    assert len(evs_dict_ex) == 1 and evs_dict_ex[0][2] == pytest.approx(2.5)
    evs_list = ed._extract_events_from_div_source([{"ExDate": ex_d, "Amount": 1.25}])
    assert len(evs_list) == 1 and evs_list[0][2] == pytest.approx(1.25)
    evs_seq_ex = ed._extract_events_from_div_source({"ExDate": [ex_d], "Amount": [1.75]})
    assert len(evs_seq_ex) == 1 and evs_seq_ex[0][2] == pytest.approx(1.75)
    merged = ed.merge_dividend_schedules(
        ["A", "B"],
        market_dividends={"A": [{"ExDate": ex_d, "Amount": 1.0}], "B": [{"ExDate": ex_d, "Amount": 2.0}]},
        predefined_dividends={"A": {"ExDates": [ex_d], "Amounts": [3.0]}},
    )
    assert merged["A"][0][2] == pytest.approx(3.0) and merged["B"][0][2] == pytest.approx(2.0)

    req_orig, _ = parsed_dag
    st = preprocess_eq_autocall(req_orig)
    assert st.canonical_codes == st.sorted_identifiers
    assert st.schedule_state is st.seasoned_state
    assert st.escrowed_dividends is st.dividend_schedule
    assert st.node_features is st.node_feats
    assert st.edge_features is st.edge_feats
    assert st.schedule_features is st.sched_feats
    assert st.global_context is st.global_feats

    jac_dict = st.surface_feature_jacobians
    uid0 = st.sorted_identifiers[0]
    assert uid0 in jac_dict
    assert len(jac_dict) == len(st.sorted_identifiers)
    assert jac_dict.get("NON_EXISTENT_ID") is None
    assert jac_dict.get(uid0) is not None
    assert list(jac_dict.keys()) == st.sorted_identifiers
    assert len(list(jac_dict.values())) == len(st.sorted_identifiers)
    assert len(list(jac_dict.items())) == len(st.sorted_identifiers)
    cloned_jac = jac_dict.clone()
    assert len(cloned_jac) == len(st.sorted_identifiers)

    asm = assemble_autocall_cashflows_and_pv(st)
    assert asm.cash_flows is asm.cashflows
    t2 = st.to_tier2_float32_tensors()
    assert all(v.dtype == torch.float32 for v in t2.values())

    # Hazard probability reconstruction (h_prob + q_prob) & already_autocalled cashflows
    import hashlib
    import math
    from deepvol.models.autocall import pre_post_processor as ppp_mod

    h_prob = torch.full((st.m_rem,), 0.10, dtype=torch.float64)
    q_prob = torch.full((st.m_rem,), 0.50, dtype=torch.float64)
    asm_hq = assemble_autocall_cashflows_and_pv(st, h_prob=h_prob, q_prob=q_prob)
    assert math.isfinite(float(asm_hq.product_value.item()))

    req_ko = copy.deepcopy(req_orig)
    req_ko.Model.MarketDataSet.AsOfDate = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    for c_k in req_ko.Model.MarketDataSet.RatesCurvesBundles.values():
        c_k.AsOfDate = req_ko.Model.MarketDataSet.AsOfDate
    for uid, ref_s in zip(req_ko.Product.UnderlyingIdentifiers, req_ko.Product.UnderlyingReferenceSpots):
        req_ko.Model.MarketDataSet.Fixings[uid].Values[3] = float(ref_s) * 1.40
    st_ko = preprocess_eq_autocall(req_ko)
    assert st_ko.already_autocalled is True
    asm_ko = assemble_autocall_cashflows_and_pv(st_ko, build_cashflow_items=True)
    assert float(asm_ko.product_value.item()) == pytest.approx(st_ko.pv_in_transit)

    # _feed_canonical_bytes edge types + FX Smooth/Linear interpolation
    h = hashlib.sha256()
    c = [0]
    assert ppp_mod._feed_canonical_bytes(h, b"abc", c) is True
    assert ppp_mod._feed_canonical_bytes(h, float("nan"), c) is True
    assert ppp_mod._feed_canonical_bytes(h, float("-0.0"), c) is True
    assert ppp_mod._feed_canonical_bytes(h, datetime.date(2026, 6, 1), c) is True
    assert ppp_mod._feed_canonical_bytes(h, [1, (2, 3), {"k": "v"}], c) is True
    assert ppp_mod._feed_canonical_bytes(h, torch.ones(2, requires_grad=True), c) is False
    fx_vol_b, fx_dcc = ed._eval_fx_atm_vol_batch(
        {
            "Tenors": ["1M", "6M", "1Y"],
            "Deltas": ["-25D", "ATM", "25D"],
            "VolatilityQuotes": [[0.18, 0.16, 0.19], [0.20, 0.18, 0.21], [0.22, 0.20, 0.23]],
            "SmileInterpolationMethod": "Smooth",
            "SmileExtrapolationMethod": "Linear",
        },
        np.array([0.05, 0.5, 2.0], dtype=np.float64),
    )
    assert fx_vol_b.shape == (3,) and fx_dcc == "Act365"

    # Non-standard FX key & reversed Corr(fx&ticker) + direct nested MarketDataSet dict bypass
    fx_res = ed._lookup_quanto_fx_spec(
        ticker="AAPL_USD",
        payment_currency="RUB",
        vol_surfaces={
            "FX_USD_RUB_CUSTOM": {"TypeName": "FXVolatilitySmileCollection"},
        },
        correlations={"FX_USD_RUB_CUSTOM&AAPL_USD": 0.35},
    )
    assert fx_res is not None and fx_res[2] == pytest.approx(0.35)

    req_raw = req_orig.model_dump()
    req_raw.pop("stripped_legacy_fields", None)
    direct_flat = {
        "Product": req_raw["Product"],
        "MarketDataSet": req_raw["Model"]["MarketDataSet"],
        "Queries": ["ProductValue"],
    }
    st_flat = preprocess_eq_autocall(direct_flat)
    assert st_flat.d_assets == len(req_orig.Product.UnderlyingIdentifiers)

    # seasoned_collapser validation & nearest-date bisect branch
    from deepvol.models.autocall.seasoned_collapser import collapse_seasoned_schedule
    as_of_c = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValueError):
        collapse_seasoned_schedule(
            as_of_date=as_of_c,
            start_date=as_of_c - datetime.timedelta(days=30),
            observation_dates=[as_of_c + datetime.timedelta(days=30)],
            payment_dates=[],
            underlying_identifiers=["A"],
            underlying_reference_spots=[100.0],
            spots_t0=[100.0],
            ko_barriers=[1.0],
            cpn_barriers=[0.75],
            fixed_coupon_amount=0.05,
        )
    with pytest.raises(ValueError):
        collapse_seasoned_schedule(
            as_of_date=as_of_c,
            start_date=as_of_c - datetime.timedelta(days=30),
            observation_dates=[as_of_c + datetime.timedelta(days=30)],
            payment_dates=[as_of_c + datetime.timedelta(days=33)],
            underlying_identifiers=["A", "B"],
            underlying_reference_spots=[100.0],
            spots_t0=[100.0, 100.0],
            ko_barriers=[1.0],
            cpn_barriers=[0.75],
            fixed_coupon_amount=0.05,
        )

    # Extra branches for _feed_canonical_bytes, collapse_seasoned_schedule & compute_escrowed_dividend_forwards
    assert ppp_mod._feed_canonical_bytes(h, torch.ones(2, dtype=torch.float64), c) is True
    assert ppp_mod._feed_canonical_bytes(h, np.ones(2, dtype=np.float64), c) is True
    assert ppp_mod._feed_canonical_bytes(h, object(), c) is False

    cs_ok = collapse_seasoned_schedule(
        as_of_date=as_of_c,
        start_date=as_of_c - datetime.timedelta(days=60),
        observation_dates=[as_of_c - datetime.timedelta(days=20), as_of_c + datetime.timedelta(days=30)],
        payment_dates=[as_of_c - datetime.timedelta(days=17), as_of_c + datetime.timedelta(days=33)],
        underlying_identifiers=["A"],
        underlying_reference_spots=[100.0],
        spots_t0=torch.tensor([95.0], dtype=torch.float32),
        ko_barriers=[1.10, 500.0],
        cpn_barriers=[],
        coupon_amounts=[0.04, 0.05],
        fixings={"A": {"Dates": [as_of_c - datetime.timedelta(days=22)], "Values": [92.0]}},
    )
    assert len(cs_ok.future_indices) == 1 and len(cs_ok.historical_records) == 1
    with pytest.raises(KeyError):
        collapse_seasoned_schedule(
            as_of_date=as_of_c,
            start_date=as_of_c - datetime.timedelta(days=60),
            observation_dates=[as_of_c - datetime.timedelta(days=20), as_of_c + datetime.timedelta(days=30)],
            payment_dates=[as_of_c - datetime.timedelta(days=17), as_of_c + datetime.timedelta(days=33)],
            underlying_identifiers=["A"],
            underlying_reference_spots=[100.0],
            spots_t0=[95.0],
            ko_barriers=[1.10, 500.0],
            cpn_barriers=[0.75, 0.75],
            fixed_coupon_amount=0.05,
            fixings={"A": {"Dates": [as_of_c - datetime.timedelta(days=18)], "Values": [92.0]}},
        )
    with pytest.raises(ValueError):
        collapse_seasoned_schedule(
            as_of_date=as_of_c,
            start_date=as_of_c - datetime.timedelta(days=30),
            observation_dates=[as_of_c + datetime.timedelta(days=30)],
            payment_dates=[as_of_c + datetime.timedelta(days=33)],
            underlying_identifiers=["A"],
            underlying_reference_spots=[100.0],
            spots_t0=[100.0],
            ko_barriers=[1.0],
            cpn_barriers=[0.75],
            fixed_coupon_amount=[0.05, 0.06],
        )
    with pytest.raises(ValueError):
        collapse_seasoned_schedule(
            as_of_date=as_of_c,
            start_date=as_of_c - datetime.timedelta(days=30),
            observation_dates=[as_of_c + datetime.timedelta(days=30)],
            payment_dates=[as_of_c + datetime.timedelta(days=33)],
            underlying_identifiers=["A"],
            underlying_reference_spots=[100.0],
            spots_t0=[100.0],
            ko_barriers=[1.0],
            cpn_barriers=[0.75],
            fixed_rate=[0.05, 0.06],
        )

    ed_out = ed.compute_escrowed_dividend_forwards(
        identifiers=["A"],
        as_of_date=as_of_c,
        observation_dates=[as_of_c + datetime.timedelta(days=180)],
        spots_t0=torch.tensor([100.0], dtype=torch.float32),
        ref_spots=torch.tensor([100.0], dtype=torch.float64),
        discount_curve=curve_obj,
        asset_financing_curves={"A": disc_dict},
        repo_curves={"A": repo_dict},
        market_dividends={"A": [{"ExDate": as_of_c + datetime.timedelta(days=30), "Amount": 2.0}]},
    )
    assert ed_out.forward_price.dtype == torch.float64 and float(ed_out.div_pv[0, 0].item()) > 1.5

    # Empty m_rem == 0, Mapping/list spots_t0 & ref_spots, past-only dividends, non-batch curves, dividend overrides
    ed_empty = ed.compute_escrowed_dividend_forwards(
        identifiers=["A"],
        as_of_date=as_of_c,
        observation_dates=[],
        spots_t0={"A": 100.0},
        ref_spots=[100.0],
        discount_curve=curve_obj,
    )
    assert ed_empty.div_pv.shape == (1, 0)

    with pytest.raises(ValueError, match="positive"):
        ed.compute_escrowed_dividend_forwards(
            identifiers=["A"],
            as_of_date=as_of_c,
            observation_dates=[as_of_c + datetime.timedelta(days=30)],
            spots_t0=[-10.0],
            ref_spots=[100.0],
            discount_curve=curve_obj,
        )

    class _CustomScalarCurve:
        def __init__(self, dcc: str) -> None:
            self.DayCountConvention = dcc

        def rate(self, dt: Any) -> float:
            return 0.05

        def df(self, dt: Any) -> float:
            return 0.98

    ed_past_and_override = ed.compute_escrowed_dividend_forwards(
        identifiers=["SBER_RUB", "APPLE_USD"],
        as_of_date=as_of_c,
        observation_dates=[as_of_c + datetime.timedelta(days=180)],
        spots_t0=[100.0, 200.0],
        ref_spots=[100.0, 200.0],
        discount_curve=_CustomScalarCurve("Act360"),
        asset_financing_curves={
            "SBER_RUB": _CustomScalarCurve("Act360"),
            "APPLE_USD": _CustomScalarCurve("Act365F"),
        },
        repo_curves={
            "SBER_RUB": _CustomScalarCurve("Act365"),
            "APPLE_USD": _CustomScalarCurve("Act365F"),
        },
        market_dividends={
            "SBER_RUB": [{"ExDate": as_of_c - datetime.timedelta(days=5), "Amount": 1.0}],
            "APPLE_USD": [{"ExDate": as_of_c - datetime.timedelta(days=5), "Amount": 1.0}],
        },
        dividend_amount_overrides={"SBER_RUB": torch.tensor([1.5], dtype=torch.float64)},
        payment_currency="RUB",
        vol_surfaces={
            "APPLE_USD": {
                "DayCountConvention": "Act365",
                "ExpirationDates": [as_of_c + datetime.timedelta(days=180)],
                "Strikes": [100.0, 200.0, 300.0],
                "VolatilityQuotes": [[0.25, 0.24, 0.25]],
            },
            "USD/RUB": {
                "TypeName": "FXVolatilitySmileCollection",
                "DayCountConvention": "Act360",
                "Deltas": ["ATM"],
                "Tenors": ["6M"],
                "VolatilityQuotes": [[0.15]],
            },
        },
        correlations={"APPLE_USD&USD/RUB": 0.35},
    )
    assert ed_past_and_override.div_pv[0, 0].item() == 0.0
    assert "APPLE_USD" in ed_past_and_override.quanto_meta

    # Cover _feed_canonical_bytes containers/SimpleNamespace, _normalize_coupon_param, _build_correlation_matrix, df overrides
    import types

    assert ppp_mod._feed_canonical_bytes(h, {1, 2}, c) is True
    assert ppp_mod._feed_canonical_bytes(h, {object()}, c) is False
    assert ppp_mod._feed_canonical_bytes(h, {"k": object()}, c) is False
    assert ppp_mod._feed_canonical_bytes(h, [object()], c) is False
    assert ppp_mod.digest_market_bytes("ns", types.SimpleNamespace(a=1)) is not None
    assert ppp_mod.digest_market_bytes("bad", torch.ones(1, requires_grad=True)) is None

    assert ppp_mod._normalize_coupon_param("0.05") == pytest.approx(0.05)
    assert ppp_mod._normalize_coupon_param(np.array([0.04, 0.05])) == [0.04, 0.05]
    assert ppp_mod._normalize_coupon_param(np.array(0.03)) == pytest.approx(0.03)
    assert ppp_mod._normalize_coupon_param([0.06]) == [0.06]

    assert ppp_mod._build_correlation_matrix(["A"], None, torch.device("cpu")).shape == (1, 1)
    assert ppp_mod._build_correlation_matrix(["A", "B"], torch.eye(2), torch.device("cpu")).shape == (2, 2)
    assert ppp_mod._build_correlation_matrix(["A", "B"], [[1.0, 0.2], [0.2, 1.0]], torch.device("cpu")).shape == (2, 2)
    with pytest.raises(ValueError):
        ppp_mod._build_correlation_matrix(["A", "B"], None, torch.device("cpu"))
    with pytest.raises(ValueError):
        ppp_mod._build_correlation_matrix(["A", "B"], {}, torch.device("cpu"))

    req_fund = copy.deepcopy(req_orig)
    req_fund.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_fund.Product.FundingSpread = 0.01
    req_fund.Product.FundingLegPaymentDates = list(req_fund.Product.PaymentDates)
    st_fund = preprocess_eq_autocall(req_fund)
    asm_ov = assemble_autocall_cashflows_and_pv(
        st_fund,
        df_fin_fn_override=lambda _d: torch.tensor(0.95, dtype=torch.float64),
        df_fwd_fn_override=lambda _d: torch.tensor(0.94, dtype=torch.float64),
    )
    assert math.isfinite(float(asm_ov.product_value.item()))

    # Missing MarketDataSet error, periods-style product payload, direct discount_curve fallback, funding fixing lookup
    with pytest.raises(ValueError, match="MarketDataSet"):
        preprocess_eq_autocall({"Product": req_raw["Product"]})

    prod_periods = types.SimpleNamespace(
        StartDate=req_orig.Product.StartDate,
        PaymentCurrency=req_orig.Product.PaymentCurrency,
        DayCountConvention="Act360",
        UnderlyingIdentifiers=list(req_orig.Product.UnderlyingIdentifiers),
        UnderlyingReferenceSpots=list(req_orig.Product.UnderlyingReferenceSpots),
        periods=[
            types.SimpleNamespace(
                observation_date=obs_dt,
                payment_date=pay_dt,
                autocall_barrier=1.10,
                coupon_barrier=0.75,
                coupon_amount=0.05,
            )
            for obs_dt, pay_dt in zip(req_orig.Product.ObservationDates, req_orig.Product.PaymentDates)
        ],
    )
    mkt_direct = copy.deepcopy(req_raw["Model"]["MarketDataSet"])
    curve_spec = next(iter(mkt_direct["RatesCurvesBundles"].values()))
    mkt_direct["discount_curve"] = curve_spec
    mkt_direct["RatesCurvesBundles"] = {}
    st_periods = preprocess_eq_autocall({"Product": prod_periods, "MarketDataSet": mkt_direct})
    assert st_periods.m_rem == st.m_rem

    assert ppp_mod._lookup_funding_fixing((), as_of_c) is None
    fix_list = [(as_of_c, 0.08), (as_of_c + datetime.timedelta(days=1, hours=3), 0.09)]
    assert ppp_mod._lookup_funding_fixing(fix_list, as_of_c) == pytest.approx(0.08)
    assert ppp_mod._lookup_funding_fixing(fix_list, as_of_c + datetime.timedelta(days=1)) == pytest.approx(0.09)
    assert ppp_mod._extract_funding_fixings({"RATE_X": {}}, "RATE_X") == ()


def test_pre_post_audit_blind_spots(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    import math
    from deepvol.models.autocall import escrowed_dividend as ed
    from deepvol.models.autocall.pre_post_processor import build_vol_surface_cache_key
    from deepvol.models.autocall.seasoned_collapser import collapse_seasoned_schedule

    req_orig, _ = parsed_dag

    # In-Transit FundingLeg across m_rem > 0, m_rem == 0, and already_autocalled=True.
    for make_m0, trigger_ko in ((False, False), (True, False), (False, True)):
        req_no_fl = copy.deepcopy(req_orig)
        as_of_fl = req_no_fl.Model.MarketDataSet.AsOfDate
        # Shift Period 11 payment slightly after AsOfDate so Period 11 is strictly In-Transit.
        req_no_fl.Product.PaymentDates[-2] = as_of_fl + datetime.timedelta(days=10)
        if make_m0:
            req_no_fl.Product.ObservationDates[-1] = as_of_fl - datetime.timedelta(days=1)
            for tkr, ref_s in zip(
                req_no_fl.Product.UnderlyingIdentifiers, req_no_fl.Product.UnderlyingReferenceSpots
            ):
                req_no_fl.Model.MarketDataSet.Fixings[tkr].Dates.append(req_no_fl.Product.ObservationDates[-1])
                req_no_fl.Model.MarketDataSet.Fixings[tkr].Values.append(float(ref_s) * 0.85)
        if trigger_ko:
            for tkr, ref_s in zip(
                req_no_fl.Product.UnderlyingIdentifiers, req_no_fl.Product.UnderlyingReferenceSpots
            ):
                req_no_fl.Model.MarketDataSet.Fixings[tkr].Values[-1] = float(ref_s) * 1.40
        req_fl = copy.deepcopy(req_no_fl)
        req_fl.Product.RateIdentifier = "RUB_RUONIA_OIS"
        req_fl.Product.FundingSpread = 0.02
        req_fl.Product.FundingLegPaymentDates = list(req_fl.Product.PaymentDates)

        st_no_fl = preprocess_eq_autocall(req_no_fl)
        st_fl = preprocess_eq_autocall(req_fl)
        asm_fl = assemble_autocall_cashflows_and_pv(st_fl)
        fund_rows = [
            row
            for row in asm_fl.cashflows
            if row.Description == "FundingLeg" and abs(float(row.Amount)) > 0.0
        ]
        assert len(fund_rows) >= 1
        assert any(float(row.Amount) < 0.0 for row in fund_rows)
        assert st_fl.pv_in_transit < st_no_fl.pv_in_transit - 1e-4
        assert math.isfinite(float(asm_fl.product_value.item()))

    # obs_date == val_date: present same-day fixing overrides spots_t0; missing fixing falls back to spots_t0.
    val_dt = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    cs_same_hit = collapse_seasoned_schedule(
        as_of_date=val_dt,
        start_date=val_dt - datetime.timedelta(days=60),
        observation_dates=[val_dt, val_dt + datetime.timedelta(days=30)],
        payment_dates=[val_dt + datetime.timedelta(days=3), val_dt + datetime.timedelta(days=33)],
        underlying_identifiers=["A"],
        underlying_reference_spots=[100.0],
        spots_t0=[90.0],
        ko_barriers=[1.10, 500.0],
        cpn_barriers=[0.75, 0.75],
        fixed_coupon_amount=0.05,
        fixings={"A": {val_dt.date().isoformat(): 120.0}},
    )
    assert cs_same_hit.already_autocalled is True

    cs_same_miss = collapse_seasoned_schedule(
        as_of_date=val_dt,
        start_date=val_dt - datetime.timedelta(days=60),
        observation_dates=[val_dt, val_dt + datetime.timedelta(days=30)],
        payment_dates=[val_dt + datetime.timedelta(days=3), val_dt + datetime.timedelta(days=33)],
        underlying_identifiers=["A"],
        underlying_reference_spots=[100.0],
        spots_t0=[90.0],
        ko_barriers=[1.10, 500.0],
        cpn_barriers=[0.75, 0.75],
        fixed_coupon_amount=0.05,
        fixings={},
    )
    assert cs_same_miss.already_autocalled is False and len(cs_same_miss.future_indices) == 1

    # Canonical 5-tier currency resolution edges.
    assert ed._infer_asset_currency("SBER_ETF", default_ccy="RUB") == "RUB"
    assert ed._infer_asset_currency("SBER_ETF", default_ccy="USD") == "USD"
    assert ed._infer_asset_currency("AAPL_US", default_ccy="RUB") == "RUB"
    assert ed._infer_asset_currency("AAPL_US", default_ccy="RUB", rate_curve_ids=("USD_SOFR",)) == "USD"
    assert ed._infer_asset_currency("AAPL_US", default_ccy="RUB", fx_smiles={"USDRUB": None}) == "USD"
    assert ed._infer_asset_currency("MUSD", default_ccy="RUB") == "RUB"

    # Surface digest and vol cache key must depend on Deltas and DayCountConvention.
    fx_s1 = {
        "Identifier": "USD/RUB",
        "TypeName": "FXVolatilitySmileCollection",
        "DayCountConvention": "Act365",
        "Deltas": ["-25D", "ATM", "+25D"],
        "Tenors": ["1M"],
        "VolatilityQuotes": [[0.20, 0.21, 0.22]],
    }
    from deepvol.api.schemas_eq_autocall import DayCountConventionEnum
    fx_s2 = {**fx_s1, "Deltas": ["-10D", "ATM", "+10D"]}
    eq_s1 = copy.deepcopy(req_orig.Model.MarketDataSet.VolatilitySurfaces[req_orig.Product.UnderlyingIdentifiers[0]])
    eq_s2 = eq_s1.model_copy(update={"DayCountConvention": DayCountConventionEnum.Act360})
    assert build_vol_surface_cache_key(fx_s1, val_dt) != build_vol_surface_cache_key(fx_s2, val_dt)
    assert build_vol_surface_cache_key(eq_s1, val_dt) != build_vol_surface_cache_key(eq_s2, val_dt)

    # Bus252 v_perf scales with surface Bus252 clock sqrt(tau_surf) instead of Act365 sqrt(tau_obs).
    from deepvol.market.eq_curves import year_fractions

    req_bus = copy.deepcopy(req_orig)
    t0_tkr = req_bus.Product.UnderlyingIdentifiers[0]
    req_bus.Model.MarketDataSet.VolatilitySurfaces[t0_tkr].DayCountConvention = DayCountConventionEnum.Business252
    st_act = preprocess_eq_autocall(req_orig)
    st_bus = preprocess_eq_autocall(req_bus)
    idx_bus = st_bus.sorted_identifiers.index(t0_tkr)
    tau_bus = year_fractions(
        req_bus.Model.MarketDataSet.AsOfDate,
        st_bus.seasoned_state.future_obs_dates,
        "Business252",
    )
    expected_v_perf_bus = st_bus.iv_at_fwd[idx_bus].detach().numpy() * np.sqrt(np.maximum(tau_bus, 1e-12))
    np.testing.assert_allclose(st_bus.v_perf[idx_bus].detach().numpy(), expected_v_perf_bus, atol=1e-12)
    assert abs(float(st_bus.v_perf[idx_bus, 0]) - float(st_act.v_perf[idx_bus, 0])) > 1e-3
    assert st_act.state_digest != st_bus.state_digest and len(st_bus.state_digest) == 64

    # Out / Down / Diff* zero coupon barrier must not become guaranteed coupon.
    norm_out = normalize_sentinel_barriers(
        [1.0, 500.0],
        [0.0, 0.0],
        coupon_barrier_type="Out",
        coupon_barrier_direction="Down",
    )
    assert float(norm_out.is_guaranteed_cpn.sum()) == 0.0

    # dividend_amount_overrides length mismatch must raise ValueError.
    with pytest.raises(ValueError):
        ed.compute_escrowed_dividend_forwards(
            identifiers=["SBER_RUB"],
            as_of_date=val_dt,
            observation_dates=[val_dt + datetime.timedelta(days=180)],
            spots_t0=[100.0],
            ref_spots=[100.0],
            discount_curve=next(iter(req_orig.Model.MarketDataSet.RatesCurvesBundles.values())),
            market_dividends={
                "SBER_RUB": [
                    {"ExDate": val_dt + datetime.timedelta(days=30), "Amount": 2.0},
                    {"ExDate": val_dt + datetime.timedelta(days=60), "Amount": 3.0},
                ]
            },
            dividend_amount_overrides={"SBER_RUB": torch.tensor([1.0], dtype=torch.float64)},
            payment_currency="RUB",
        )

    # Scalar custom curve with m_rem=2 and future dividends.
    class _ScalarCurve2M:
        DayCountConvention = "Act365"

        def rate(self, _dt: Any) -> float:
            return 0.06

        def df(self, _dt: Any) -> float:
            return 0.97

    ed_2m = ed.compute_escrowed_dividend_forwards(
        identifiers=["SBER_RUB"],
        as_of_date=val_dt,
        observation_dates=[val_dt + datetime.timedelta(days=90), val_dt + datetime.timedelta(days=180)],
        spots_t0=[100.0],
        ref_spots=[100.0],
        discount_curve=_ScalarCurve2M(),
        market_dividends={"SBER_RUB": [{"ExDate": val_dt + datetime.timedelta(days=45), "Amount": 2.0}]},
        payment_currency="RUB",
    )
    assert ed_2m.forward_perf.shape == (1, 2) and torch.isfinite(ed_2m.forward_perf).all()


def test_settles_today_ki_obs_is_long_basket_dual_expiry_and_coupon_undisc(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify SettlesToday PV inclusion, ki_obs_type persistence, IsLongBasket option_index,
    AllowAutocallOnExpiry dual maturity Notional rows, and Class B2 coupon_cf_undisc override."""
    req_orig, _ = parsed_dag

    # 1. SettlesToday with remaining future periods: past < AsOfDate is Settled, == AsOfDate is SettlesToday,
    #    and cash_to_settle_today is included in ProductValue on both fast and AD paths.
    req_st = copy.deepcopy(req_orig)
    as_of_st = req_st.Product.PaymentDates[9]
    req_st.Product.IsMemory = False
    req_st.Model.MarketDataSet.AsOfDate = as_of_st
    for curve in req_st.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = as_of_st
    for tkr, ref_s in zip(req_st.Product.UnderlyingIdentifiers, req_st.Product.UnderlyingReferenceSpots):
        fix = req_st.Model.MarketDataSet.Fixings[tkr]
        # Force period 9 (index 9) coupon hit (above 0.90 cpn barrier, below 0.925 KO barrier)
        fix.Values[9] = float(ref_s) * 0.91
    st_today = preprocess_eq_autocall(req_st)
    assert st_today.already_autocalled is False and st_today.m_rem >= 1
    assert isinstance(req_st.Product.FixedCouponAmount, float)
    assert st_today.cash_to_settle_today == pytest.approx(float(req_st.Product.FixedCouponAmount), abs=1e-12)
    settles_today_rows = [cf for cf in st_today.seasoned_state.past_cashflows if cf.StatusText == "SettlesToday"]
    assert len(settles_today_rows) == 3
    assert all(cf.Date.date() == as_of_st.date() for cf in settles_today_rows)

    p0 = torch.zeros(st_today.m_rem, dtype=torch.float64)
    q0 = torch.zeros(st_today.m_rem, dtype=torch.float64)
    asm_fast = assemble_autocall_cashflows_and_pv(st_today, p_stop=p0, q_cpn=q0, l_put=torch.tensor(0.0, dtype=torch.float64))
    asm_ad = assemble_autocall_cashflows_and_pv(
        st_today,
        p_stop=p0.clone().requires_grad_(True),
        q_cpn=q0.clone().requires_grad_(True),
        l_put=torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
    )
    expected_pv_today = float(asm_fast.notional_pv.item()) + float(asm_fast.coupon_pv.item()) + st_today.cash_to_settle_today
    assert float(asm_fast.product_value.item()) == pytest.approx(expected_pv_today, abs=1e-12)
    assert float(asm_ad.product_value.item()) == pytest.approx(expected_pv_today, abs=1e-12)

    # 2. ki_obs_type persistence across SeasonedScheduleState, AutocallPreprocState, digest, and spot template
    digests: dict[str | None, str] = {}
    for mode in (None, "European", "Discrete", "Daily", "Continuous"):
        req_ki = copy.deepcopy(req_orig)
        req_ki.Product.OptionDownAndInBarrierLevel = 0.70
        req_ki.Product.OptionDownAndInBarrierObservationType = mode
        st_ki = preprocess_eq_autocall(req_ki)
        assert st_ki.ki_obs_type == mode
        assert st_ki.seasoned_state.ki_obs_type == mode
        digests[mode] = st_ki.state_digest
        # Spot-shock template reuse preserves ki_obs_type
        req_ki_bump = copy.deepcopy(req_ki)
        t0_k = req_ki_bump.Product.UnderlyingIdentifiers[0]
        req_ki_bump.Model.MarketDataSet.Spots[t0_k] *= 1.01
        st_ki_bump = preprocess_eq_autocall(req_ki_bump)
        assert st_ki_bump.ki_obs_type == mode
        assert st_ki_bump.seasoned_state.ki_obs_type == mode
    assert len(set(digests.values())) == 5

    # 3. IsLongBasket preserves true option_index (WorstOf) instead of forcing AverageOf
    req_lb = copy.deepcopy(req_orig)
    req_lb.Product.IsLongBasket = True
    req_lb.Product.OptionMonitoringIndex = "WorstOf"
    req_lb.Product.OptionDownAndInBarrierLevel = 0.80
    req_lb.Product.OptionDownAndInBarrierObservationType = "Discrete"
    # Breach WorstOf in historical period 0 while AverageOf stays above 0.80
    u0 = req_lb.Product.UnderlyingIdentifiers[0]
    req_lb.Model.MarketDataSet.Fixings[u0].Values[0] = float(req_lb.Product.UnderlyingReferenceSpots[0]) * 0.50
    for u_other, ref_other in zip(
        req_lb.Product.UnderlyingIdentifiers[1:], req_lb.Product.UnderlyingReferenceSpots[1:]
    ):
        req_lb.Model.MarketDataSet.Fixings[u_other].Values[0] = float(ref_other) * 1.10
    st_lb = preprocess_eq_autocall(req_lb)
    assert st_lb.is_long_basket is True
    assert st_lb.option_index == "WorstOf"
    assert st_lb.seasoned_state.i_ki_past == pytest.approx(1.0, abs=1e-12)

    # 4. AllowAutocallOnExpiry=True emits dual maturity Notional rows (expiry-autocall + survival)
    req_exp = copy.deepcopy(req_orig)
    req_exp.Product.AllowAutocallOnExpiry = True
    req_exp.Product.AutoCallTriggerLevels[-1] = 1.00
    st_exp = preprocess_eq_autocall(req_exp)
    asm_exp = assemble_autocall_cashflows_and_pv(
        st_exp,
        p_stop=torch.tensor([0.25], dtype=torch.float64),
        q_cpn=torch.tensor([0.25], dtype=torch.float64),
        l_put=torch.tensor(0.10, dtype=torch.float64),
    )
    assert len(asm_exp.cashflows) == 38
    df_last = float(st_exp.df_pay[-1].item())
    tail_rows = asm_exp.cashflows[-5:]
    assert [r.Description for r in tail_rows] == ["FundingLeg", "Coupon", "Notional", "Option", "Notional"]
    assert tail_rows[2].Amount == pytest.approx(df_last * 0.25, abs=1e-12)
    assert tail_rows[4].Amount == pytest.approx(df_last * 0.75, abs=1e-12)
    assert tail_rows[2].Amount + tail_rows[4].Amount == pytest.approx(float(asm_exp.notional_pv.item()), abs=1e-12)

    # 5. Class B2 coupon_cf_undisc is always honored for IsMemory=Y (fast + AD paths)
    req_mem = copy.deepcopy(req_orig)
    req_mem.Model.MarketDataSet.AsOfDate = req_mem.Product.StartDate
    for curve in req_mem.Model.MarketDataSet.RatesCurvesBundles.values():
        curve.AsOfDate = req_mem.Product.StartDate
    st_mem = preprocess_eq_autocall(req_mem)
    assert st_mem.is_memory is True and st_mem.m_rem > 1
    undisc_override = np.linspace(0.01, 0.05, st_mem.m_rem, dtype=np.float64)
    p_mem = torch.full((st_mem.m_rem,), 0.02, dtype=torch.float64)
    q_mem = torch.full((st_mem.m_rem,), 0.50, dtype=torch.float64)
    asm_b2_fast = assemble_autocall_cashflows_and_pv(
        st_mem, p_stop=p_mem, q_cpn=q_mem, coupon_cf_undisc=undisc_override
    )
    undisc_t = torch.tensor(undisc_override, dtype=torch.float64, requires_grad=True)
    asm_b2_ad = assemble_autocall_cashflows_and_pv(
        st_mem, p_stop=p_mem, q_cpn=q_mem, coupon_cf_undisc=undisc_t
    )
    expected_cpn_pv = float((st_mem.df_pay * torch.tensor(undisc_override, dtype=torch.float64)).sum().item())
    assert float(asm_b2_fast.coupon_pv.item()) == pytest.approx(expected_cpn_pv, abs=1e-12)
    assert float(asm_b2_ad.coupon_pv.item()) == pytest.approx(expected_cpn_pv, abs=1e-12)
    asm_b2_ad.coupon_pv.backward()
    assert undisc_t.grad is not None
    torch.testing.assert_close(undisc_t.grad, st_mem.df_pay, atol=1e-12, rtol=1e-12)










