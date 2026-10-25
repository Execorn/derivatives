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
REF_REQUEST_PATH = REPO_ROOT / "eq_autocallable_request.txt"


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
