from __future__ import annotations

import copy
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError
import pytest
import torch

from deepvol.api.eq_autocall_arrow import ARROW_MIME, encode_eq_autocall_arrow_request
from deepvol.api.eq_autocall_runtime import ALL_8_GREEK_QUERIES, EQAutocallRuntime
from deepvol.api.schemas_eq_autocall import (
    CurveInterpolationMethodEnum,
    DividendItemSpec,
    EQAutocallableTradeRequest,
    MonitoringIndexEnum,
    parse_murex_autocall_dag,
)
from deepvol.api.server import app
from deepvol.calibration.autocall_sobol_generator import generate_synthetic_autocall_batch
from deepvol.market.eq_curves import ClampedYieldCurve, RepoCurve
from deepvol.models.autocall import analytical_baseline as ab
from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig, DupireQMCGpuSolver
from deepvol.models.autocall.feature_layout import MonitoringIndexCode, monitoring_index_to_code
from deepvol.models.autocall.greeks_engine import (
    compute_eq_autocall_greeks,
    compute_qmc_crn_greeks,
    evaluate_differentiable_pv,
)
from deepvol.models.autocall.pre_post_processor import (
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.seasoned_collapser import collapse_seasoned_schedule
from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate
from deepvol.surrogates.order_statistic_pooling import OrderStatisticPooling

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


def _load_base_request() -> EQAutocallableTradeRequest:
    req, _ = parse_murex_autocall_dag(REF_REQUEST_PATH)
    return req


def _slice_assets(req: EQAutocallableTradeRequest, n_assets: int) -> EQAutocallableTradeRequest:
    out = copy.deepcopy(req)
    tks = out.Product.UnderlyingIdentifiers[:n_assets]
    out.Product.UnderlyingIdentifiers = tks
    out.Product.UnderlyingReferenceSpots = out.Product.UnderlyingReferenceSpots[:n_assets]
    out.Model.MarketDataSet.Spots = {k: out.Model.MarketDataSet.Spots[k] for k in tks}
    out.Model.MarketDataSet.VolatilitySurfaces = {
        k: out.Model.MarketDataSet.VolatilitySurfaces[k] for k in tks
    }
    out.Model.MarketDataSet.Dividends = {
        k: out.Model.MarketDataSet.Dividends[k]
        for k in tks
        if k in out.Model.MarketDataSet.Dividends
    }
    out.Product.PredefinedDividends = {
        k: out.Product.PredefinedDividends[k]
        for k in tks
        if out.Product.PredefinedDividends and k in out.Product.PredefinedDividends
    }
    out.Model.MarketDataSet.Fixings = {
        k: out.Model.MarketDataSet.Fixings[k]
        for k in tks
        if k in out.Model.MarketDataSet.Fixings
    }
    out.Model.MarketDataSet.Correlations = {
        f"{tks[i]}&{tks[j]}": 0.45 for i in range(n_assets) for j in range(i + 1, n_assets)
    }
    return out


def _build_dim_horizon_request(
    base: EQAutocallableTradeRequest,
    *,
    d_assets: int,
    m_periods: int,
) -> EQAutocallableTradeRequest:
    req = copy.deepcopy(base)
    as_of = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    req.Model.MarketDataSet.AsOfDate = as_of
    for c in req.Model.MarketDataSet.RatesCurvesBundles.values():
        c.AsOfDate = as_of

    tks = [f"EQ{i:02d}_RUB" for i in range(d_assets)]
    tpl_surf = next(iter(req.Model.MarketDataSet.VolatilitySurfaces.values()))
    req.Product.StartDate = as_of
    req.Product.ObservationDates = [as_of + datetime.timedelta(days=30 * (k + 1)) for k in range(m_periods)]
    req.Product.PaymentDates = [d + datetime.timedelta(days=3) for d in req.Product.ObservationDates]
    req.Product.AutoCallTriggerLevels = [1.05] * max(m_periods - 1, 0) + ([500.0] if m_periods > 0 else [])
    req.Product.CouponBarrierLevels = [0.75] * m_periods
    req.Product.FixedCouponAmount = 0.03
    req.Product.FixedRate = None
    req.Product.UnderlyingIdentifiers = tks
    req.Product.UnderlyingReferenceSpots = [100.0] * d_assets
    req.Product.PredefinedDividends = {}
    req.Model.MarketDataSet.Spots = {tk: 92.0 + float(i % 5) * 3.0 for i, tk in enumerate(tks)}
    req.Model.MarketDataSet.VolatilitySurfaces = {
        tk: tpl_surf.model_copy(update={"Identifier": tk, "AsOfDate": as_of}) for tk in tks
    }
    req.Model.MarketDataSet.Dividends = {}
    req.Model.MarketDataSet.Fixings = {}
    req.Model.MarketDataSet.Correlations = {
        f"{tks[i]}&{tks[j]}": 0.4 for i in range(d_assets) for j in range(i + 1, d_assets)
    }
    return req


def test_lfi_path_dos_explicit_none_and_fixing_coercion() -> None:
    for bad_path in ("/etc/passwd", "x" * 260):
        with pytest.raises(ValueError):
            parse_murex_autocall_dag(bad_path)
        with TestClient(app) as cli:
            r_lfi = cli.post(
                "/eq_autocall/price_murex_dag",
                content=bad_path.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
            )
            assert r_lfi.status_code == 422

    req = _load_base_request()
    base_d = req.model_dump(mode="json")
    base_d.pop("stripped_legacy_fields", None)
    tkr0 = base_d["Product"]["UnderlyingIdentifiers"][0]
    obs_iso = str(base_d["Product"]["ObservationDates"][0])

    # Explicit None optional defaults on 1-asset payload + ticker-keyed and date-keyed dict fixings.
    d_opt = copy.deepcopy(_slice_assets(req, 1).model_dump(mode="json"))
    d_opt.pop("stripped_legacy_fields", None)
    d_opt["Product"]["DayCountConvention"] = None
    d_opt["Product"]["CallabilityMonitoringIndex"] = None
    d_opt["Product"]["PredefinedDividends"] = None
    d_opt["Model"]["TypeName"] = None
    d_opt["Model"]["FinancingCurveIdentifiers"] = None
    d_opt["Model"]["MarketDataSet"]["Correlations"] = None
    d_opt["Model"]["MarketDataSet"]["RepoSpreadCurves"] = None
    d_opt["Model"]["MarketDataSet"]["Dividends"] = None
    d_opt["Model"]["MarketDataSet"]["Fixings"] = {tkr0: {"Dates": [obs_iso], "Values": [98.0]}}
    d_opt["stripped_legacy_fields"] = None
    parsed_opt = EQAutocallableTradeRequest.model_validate(d_opt)
    assert parsed_opt.Model.MarketDataSet.Fixings[tkr0].Identifier == tkr0

    val_dt_fix = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    cs_flat = collapse_seasoned_schedule(
        as_of_date=val_dt_fix,
        start_date=val_dt_fix - datetime.timedelta(days=60),
        observation_dates=[val_dt_fix - datetime.timedelta(days=5), val_dt_fix + datetime.timedelta(days=30)],
        payment_dates=[val_dt_fix - datetime.timedelta(days=2), val_dt_fix + datetime.timedelta(days=33)],
        underlying_identifiers=[tkr0],
        underlying_reference_spots=[100.0],
        spots_t0=[90.0],
        ko_barriers=[0.95, 1.15],
        cpn_barriers=[0.80, 0.80],
        fixed_coupon_amount=0.04,
        fixings={tkr0: {(val_dt_fix - datetime.timedelta(days=5)).date().isoformat(): 98.0}},
    )
    assert cs_flat.already_autocalled is True

    dt_ex = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    dt_pay = datetime.datetime(2026, 6, 5, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValidationError):
        DividendItemSpec(ExDate=dt_ex, PaymentDate=dt_pay, Amount=-1.0)
    for bad_num in (float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            DividendItemSpec(ExDate=dt_ex, PaymentDate=dt_pay, Amount=bad_num)


def test_http_query_flags_currency_arrow_m0_and_413() -> None:
    req = _load_base_request()
    base_d = req.model_dump(mode="json")
    base_d.pop("stripped_legacy_fields", None)
    tkr0 = base_d["Product"]["UnderlyingIdentifiers"][0]
    base_d["Model"]["MarketDataSet"]["VolatilitySurfaces"][tkr0]["Currency"] = "EUR"
    if tkr0 in base_d["Model"]["MarketDataSet"].get("Dividends", {}):
        base_d["Model"]["MarketDataSet"]["Dividends"][tkr0]["Currency"] = "EUR"
    base_d["Product"]["PredefinedDividends"] = {
        tkr0: {
            "Currency": "EUR",
            "ExDates": ["2026-12-01"],
            "Amounts": [2.0],
        }
    }
    dumped = EQAutocallableTradeRequest.model_validate(base_d).model_dump(mode="json")
    assert dumped["Model"]["MarketDataSet"]["VolatilitySurfaces"][tkr0]["Currency"] == "EUR"
    assert dumped["Product"]["PredefinedDividends"][tkr0]["Currency"] == "EUR"
    dumped_pv = copy.deepcopy(dumped)
    dumped_pv["Queries"] = ["ProductValue", "CashFlows"]
    dumped_grk = copy.deepcopy(dumped)
    dumped_grk["Queries"] = ["ProductValue", "Delta"]

    with TestClient(app) as cli:
        r_yes = cli.post("/eq_autocall/price?include_all_greeks=true", json=dumped_pv)
        assert r_yes.status_code == 200 and r_yes.json()["Delta"] is not None
        r_no = cli.post("/eq_autocall/price?include_all_greeks=false", json=dumped_pv)
        assert r_no.status_code == 200 and r_no.json()["Delta"] is None
        r_bad_bool = cli.post("/eq_autocall/price?include_all_greeks=invalid_bool", json=dumped_pv)
        assert r_bad_bool.status_code == 422
        r_b_no = cli.post("/eq_autocall/price_batch?compute_greeks=false", json=[dumped_grk])
        assert r_b_no.status_code == 200 and r_b_no.json()[0]["Delta"] is None
        assert cli.post("/eq_autocall/price_batch?compute_greeks=invalid_bool", json=[dumped_grk]).status_code == 422
        assert (
            cli.post("/eq_autocall/price_batch", json={"requests": [dumped_grk], "compute_greeks": "false"}).status_code
            == 422
        )

        oversized = b"{" + (b" " * (50 * 1024 * 1024 + 16)) + b"}"
        assert (
            cli.post(
                "/eq_autocall/price",
                content=oversized,
                headers={"Content-Type": "application/json"},
            ).status_code
            == 413
        )

        synth = generate_synthetic_autocall_batch(
            batch_size=2, d_max=8, m_max=48, n_mc_paths=32, seed=101, device="cpu", dtype=torch.float64
        )
        arrow_m0 = encode_eq_autocall_arrow_request(
            synth.node_feats,
            synth.edge_feats,
            synth.asset_mask,
            synth.sched_feats,
            torch.zeros_like(synth.sched_mask),
            synth.global_feats,
            pv_in_transit=torch.tensor([0.05, 0.10], dtype=torch.float64),
        )
        r_m0 = cli.post(
            "/eq_autocall/price_arrow?compute_greeks=true",
            content=arrow_m0,
            headers={"Content-Type": ARROW_MIME},
        )
        assert r_m0.status_code == 200


def test_in_transit_funding_obs_eq_val_and_curves() -> None:
    req = _load_base_request()
    req_fl = copy.deepcopy(req)
    req_fl.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_fl.Product.FundingSpread = 0.02
    req_fl.Product.FundingLegPaymentDates = list(req_fl.Product.PaymentDates)
    as_of = req_fl.Model.MarketDataSet.AsOfDate
    req_fl.Product.ObservationDates[-1] = as_of - datetime.timedelta(days=1)
    req_fl.Product.PaymentDates[-2] = as_of + datetime.timedelta(days=10)
    req_fl.Product.FundingLegPaymentDates[-2] = as_of + datetime.timedelta(days=10)
    for tkr, ref_s in zip(req_fl.Product.UnderlyingIdentifiers, req_fl.Product.UnderlyingReferenceSpots):
        req_fl.Model.MarketDataSet.Fixings[tkr].Dates.append(req_fl.Product.ObservationDates[-1])
        req_fl.Model.MarketDataSet.Fixings[tkr].Values.append(float(ref_s) * 0.85)
    st_fl0 = preprocess_eq_autocall(req_fl)
    asm_fl0 = assemble_autocall_cashflows_and_pv(st_fl0)
    assert st_fl0.m_rem == 0
    assert any(r.Description == "FundingLeg" and float(r.Amount) < 0.0 for r in asm_fl0.cashflows)
    assert math.isfinite(float(asm_fl0.product_value.item()))

    # obs == val with and without same-day fixing.
    val_dt = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    cs_hit = collapse_seasoned_schedule(
        as_of_date=val_dt,
        start_date=val_dt - datetime.timedelta(days=30),
        observation_dates=[val_dt, val_dt + datetime.timedelta(days=30)],
        payment_dates=[val_dt + datetime.timedelta(days=2), val_dt + datetime.timedelta(days=32)],
        underlying_identifiers=["A"],
        underlying_reference_spots=[100.0],
        spots_t0=[90.0],
        ko_barriers=[1.10, 500.0],
        cpn_barriers=[0.75, 0.75],
        fixed_coupon_amount=0.05,
        fixings={"A": {val_dt.date().isoformat(): 125.0}},
    )
    assert cs_hit.already_autocalled is True

    cs_miss = collapse_seasoned_schedule(
        as_of_date=val_dt,
        start_date=val_dt - datetime.timedelta(days=30),
        observation_dates=[val_dt, val_dt + datetime.timedelta(days=30)],
        payment_dates=[val_dt + datetime.timedelta(days=2), val_dt + datetime.timedelta(days=32)],
        underlying_identifiers=["A"],
        underlying_reference_spots=[100.0],
        spots_t0=[90.0],
        ko_barriers=[1.10, 500.0],
        cpn_barriers=[0.75, 0.75],
        fixed_coupon_amount=0.05,
        fixings={},
    )
    assert cs_miss.already_autocalled is False and len(cs_miss.future_indices) == 1

    # Curves: space_coords, MonotoneConvex C^0 continuity, Smooth, RepoCurve edges.
    t_grid = np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    r_pct = np.array([12.0, 12.6, 13.1, 13.5], dtype=np.float64)
    c_rate = ClampedYieldCurve(as_of_date=val_dt, pillar_tenors=t_grid, rates=r_pct, interpolation_method="LogLinear")
    c_logdf = ClampedYieldCurve(
        as_of_date=val_dt,
        pillar_tenors=t_grid,
        rates=-(r_pct / 100.0) * t_grid,
        interpolation_method="LogLinear",
        space_coords="LogarithmOfDiscountFactors",
        quote_expression="Standard",
    )
    np.testing.assert_allclose(c_logdf.df([0.3, 1.5]), c_rate.df([0.3, 1.5]), atol=1e-12)
    c_sm = ClampedYieldCurve(
        as_of_date=val_dt,
        pillar_tenors=t_grid,
        rates=r_pct,
        interpolation_method=CurveInterpolationMethodEnum.Smooth,
    )
    assert np.all(np.isfinite(c_sm.df([0.1, 0.8, 3.0])))
    for stress_r in ([1.0, 2.0, 3.0, 6.0], [5.0, 5.0, 6.0, 30.0]):
        c_mc = ClampedYieldCurve(
            as_of_date=val_dt, pillar_tenors=t_grid, rates=stress_r, interpolation_method="MonotoneConvex"
        )
        for tk in t_grid[1:-1]:
            f_l = -(math.log(float(c_mc.df(tk))) - math.log(float(c_mc.df(tk - 1e-6)))) / 1e-6
            f_r = -(math.log(float(c_mc.df(tk + 1e-6))) - math.log(float(c_mc.df(tk)))) / 1e-6
            assert abs(f_r - f_l) < 1e-3
    with pytest.raises(ValueError, match="repo period"):
        RepoCurve(
            {"Identifier": "REPO_NEG", "AsOfDate": val_dt, "RatesSchedule": {"Periods": ["-30D", "1Y"], "Rates": [0.5, 0.8]}}
        )
    rc_edge = RepoCurve(
        {"Identifier": "REPO_E", "AsOfDate": val_dt, "RatesSchedule": {"Periods": ["30D", "1Y"], "Rates": [0.5, 0.8]}}
    )
    assert np.isfinite(float(rc_edge.repo_rate(np.int64(90) / 365.0)))


def test_exotic_payoffs_18_indices_dimensions_and_all8_greeks() -> None:
    # DiffAB k_strike=0 uncapped normal put, TwinWin, OneStar d=1, ForceCouponsOnAutocall.
    p_diff, fw_d, vw_d, _ = ab.fast_stulz_margrabe_cpu(
        [1.02, 0.96], [0.25, 0.30], [[1.0, 0.4], [0.4, 1.0]], k_strike=0.0, monitoring_index="DiffAB"
    )
    s_d = max(vw_d, 1e-12)
    z_d = -fw_d / s_d
    exp_d = -fw_d * float(ab._ndtr_scalar(z_d)) + s_d * float(ab._INV_SQRT2PI * math.exp(-0.5 * z_d * z_d))
    assert p_diff == pytest.approx(exp_d, abs=1e-12) and p_diff > 0.0

    os_out = ab.compute_analytical_autocall_baseline(
        {
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
            "one_star_strike": 1.10,
            "one_star_coupon": 0.05,
            "no_notional_repayment": True,
        },
        use_fast_cpu=True,
    )
    assert float(os_out.l_put_proxy.item()) == pytest.approx(0.40, abs=0.02)

    tw_out = ab.compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.tensor([[1.02]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.20]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
            "df_pay": torch.tensor([0.96], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.30], dtype=torch.float64),
            "is_callable": torch.tensor([1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([0.70], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.03], dtype=torch.float64),
            "option_index": "TwinWin",
            "allow_autocall_on_expiry": True,
        },
        use_fast_cpu=False,
    )
    assert torch.isfinite(tw_out.product_value_proxy).all()

    fc_out = ab.compute_analytical_autocall_baseline(
        {
            "forward_perf": torch.tensor([[1.00], [1.00]], dtype=torch.float64),
            "v_perf": torch.tensor([[0.25], [0.35]], dtype=torch.float64),
            "corr_matrix": torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
            "df_pay": torch.tensor([0.96], dtype=torch.float64),
            "ko_barriers": torch.tensor([1.05], dtype=torch.float64),
            "is_callable": torch.tensor([1.0], dtype=torch.float64),
            "cpn_barriers": torch.tensor([1.02], dtype=torch.float64),
            "is_guaranteed_cpn": torch.tensor([0.0], dtype=torch.float64),
            "coupon_amounts": torch.tensor([0.05], dtype=torch.float64),
            "coupon_barrier_type": "Out",
            "coupon_barrier_direction": "Down",
            "force_coupons_on_autocall": True,
        },
        use_fast_cpu=True,
    )
    assert torch.isfinite(fc_out.q_proxy).all()

    # All 18 index codes (0..17) + aliases + 15/18 pooling compatibility.
    for idx_enum in MonitoringIndexEnum:
        code = monitoring_index_to_code(idx_enum.value)
        assert 0 <= code <= 17
    assert int(MonitoringIndexCode.SECOND_WORST_OF) == 15
    assert int(MonitoringIndexCode.THIRD_WORST_OF) == 16
    assert int(MonitoringIndexCode.SECOND_BEST_OF) == 17
    assert monitoring_index_to_code("EquallyWeightedBasket") == int(MonitoringIndexCode.AVERAGE_OF)

    pool18 = OrderStatisticPooling(hidden_dim=16, out_dim=32, n_func_codes=18, func_emb_dim=8)
    pool15 = OrderStatisticPooling(hidden_dim=16, out_dim=32, n_func_codes=15, func_emb_dim=8)
    pool18.load_state_dict(pool15.state_dict(), strict=False)
    for c_i in range(18):
        out_p = pool18(
            torch.randn(1, 5, 16),
            torch.ones(1, 5),
            func_codes=torch.full((1, 3), c_i, dtype=torch.long),
        )
        assert torch.isfinite(out_p).all()

    # d=1..12, M=1..60, All-8 Greeks, standalone AggCashGamma, and non-zero std=1e-3 surrogate check.
    base = _load_base_request()
    for d_dim, m_len in ((1, 1), (4, 12), (9, 24), (12, 60)):
        st_dm = preprocess_eq_autocall(_build_dim_horizon_request(base, d_assets=d_dim, m_periods=m_len), d_max=12, m_max=60)
        assert st_dm.d_assets == d_dim and st_dm.m_rem == m_len
        assert st_dm.baseline is not None and torch.isfinite(st_dm.baseline.product_value_proxy).all()

    req2 = _slice_assets(base, 2)
    st2 = preprocess_eq_autocall(req2)
    cfg2 = DupireQMCConfig(n_paths=256, n_replicates=4, n_substeps=2)
    qmc_all8 = compute_qmc_crn_greeks(st2, queries=list(ALL_8_GREEK_QUERIES), config=cfg2)
    qmc_cg_only = compute_qmc_crn_greeks(
        st2, queries=["ProductValue", "MultivariateAggregatedCashGamma"], config=cfg2
    )
    assert qmc_all8.delta is not None and qmc_all8.gamma is not None and qmc_all8.theta is not None
    assert qmc_all8.multivariate_aggregated_cash_gamma is not None
    assert qmc_cg_only.multivariate_aggregated_cash_gamma is not None
    assert float(qmc_cg_only.multivariate_aggregated_cash_gamma) == pytest.approx(
        float(qmc_all8.multivariate_aggregated_cash_gamma), abs=1e-4, rel=1e-2
    )

    surr32 = MaskedEQAutocallSurrogate(d_max=8, m_max=48, d_model=64, n_egno_layers=2, n_sched_layers=1).eval()
    surr32.zero_init_residual_heads(std=1e-3)
    assert not surr32.is_zero_residual()
    res_surr = compute_eq_autocall_greeks(st2, queries=["ProductValue", "Delta"], surrogate=surr32)
    sp0 = float(st2.spots_t0[0].item())
    h0 = sp0 * 1e-3
    sp_u, sp_d = st2.spots_t0.detach().clone(), st2.spots_t0.detach().clone()
    sp_u[0] += h0
    sp_d[0] -= h0
    out_u = evaluate_differentiable_pv(st2, spots_t0=sp_u, surrogate=surr32)
    out_d = evaluate_differentiable_pv(st2, spots_t0=sp_d, surrogate=surr32)
    assert isinstance(out_u, torch.Tensor) and isinstance(out_d, torch.Tensor)
    fd0 = (float(out_u.item()) - float(out_d.item())) / (2.0 * h0)
    assert res_surr.delta_tensor is not None
    got0 = float(res_surr.delta_tensor[st2.inv_lex_perm[0]].item())
    assert abs(got0 - fd0) / max(abs(fd0), 1e-10) <= 1e-5


def _audit_group_cashflows(cf_rows: object) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], float] = {}
    if not isinstance(cf_rows, (list, tuple)):
        return grouped
    for row in cf_rows:
        dt_val = row.get("Date") if isinstance(row, dict) else getattr(row, "Date", "")
        dt_str = (
            dt_val.strftime("%Y-%m-%d")
            if isinstance(dt_val, (datetime.datetime, datetime.date))
            else str(dt_val)[:10]
        )
        desc = str(row.get("Description") if isinstance(row, dict) else getattr(row, "Description", ""))
        amt = float(row.get("Amount", 0.0) if isinstance(row, dict) else getattr(row, "Amount", 0.0))
        grouped[(dt_str, desc)] = grouped.get((dt_str, desc), 0.0) + amt
    return grouped


def _audit_assert_dual_gate_oracle(
    tid: str,
    rt_res: object,
    oracle: object,
    mf_pv: float,
    state: object,
    req_model: EQAutocallableTradeRequest,
) -> None:
    from deepvol.api.eq_autocall_runtime import EQAutocallRuntimeResponse, _merge_fx_vega_and_cega
    from deepvol.api.schemas_eq_autocall import to_decimal_merged_rho
    from deepvol.models.autocall.greeks_engine import AutocallGreeksResult
    from deepvol.models.autocall.pre_post_processor import AutocallPreprocState

    assert isinstance(rt_res, EQAutocallRuntimeResponse)
    assert isinstance(oracle, AutocallGreeksResult)
    assert isinstance(state, AutocallPreprocState)

    spots_map = {str(k): float(v) for k, v in req_model.Model.MarketDataSet.Spots.items()}
    assert abs(rt_res.ProductValue - float(oracle.product_value)) * 1e4 < 5.0
    assert abs(rt_res.ProductValue - mf_pv) * 1e4 < 5.0

    rt_cf = _audit_group_cashflows(rt_res.CashFlows)
    or_cf = _audit_group_cashflows(oracle.cash_flows)
    for k in sorted(set(rt_cf) | set(or_cf)):
        assert abs(rt_cf.get(k, 0.0) - or_cf.get(k, 0.0)) * 1e4 < 5.0

    assert rt_res.Delta is not None and oracle.delta is not None
    assert set(rt_res.Delta.keys()) == set(oracle.delta.keys())
    for tk in rt_res.Delta:
        s_tk = float(spots_map.get(tk, 1.0))
        assert abs((float(rt_res.Delta[tk]) - float(oracle.delta[tk])) * s_tk) * 0.01 * 1e4 < 5.0

    assert isinstance(rt_res.Gamma, dict) and oracle.gamma is not None
    for a, row_or in oracle.gamma.items():
        s_a = float(spots_map.get(a, 1.0))
        row_rt = rt_res.Gamma[a]
        assert isinstance(row_rt, dict)
        for b, ov in row_or.items():
            s_b = float(spots_map.get(b, 1.0))
            rv = float(row_rt[b])
            assert 0.5 * (0.01**2) * abs((rv - float(ov)) * s_a * s_b) * 1e4 < 5.0

    assert isinstance(rt_res.MultivariateAggregatedCashGamma, (int, float))
    assert oracle.multivariate_aggregated_cash_gamma is not None
    assert (
        0.5
        * (0.01**2)
        * abs(float(rt_res.MultivariateAggregatedCashGamma) - float(oracle.multivariate_aggregated_cash_gamma))
        * 1e4
        < 5.0
    )

    or_vega, or_cega = _merge_fx_vega_and_cega(oracle, state, req_model)
    assert rt_res.Vega is not None and or_vega is not None
    for tk, ov_grid in or_vega.items():
        rt_vg = np.asarray(rt_res.Vega[tk], dtype=np.float64)
        or_vg = np.asarray(ov_grid, dtype=np.float64)
        assert abs(float(np.sum(rt_vg)) - float(np.sum(or_vg))) * 0.01 * 1e4 < 5.0
        if rt_vg.ndim == 2 and rt_vg.shape == or_vg.shape:
            assert float(np.max(np.abs(np.sum(rt_vg, axis=1) - np.sum(or_vg, axis=1)))) * 0.01 * 1e4 < 5.0

    rate_scales = {
        str(cid): (0.01 if "Percentage" in str(getattr(c_spec, "RateQuoteExpression", "Percentage")) else 1.0)
        for cid, c_spec in (req_model.Model.MarketDataSet.RatesCurvesBundles or {}).items()
    }
    repo_scales = {
        str(tk): (0.01 if "Percentage" in str(getattr(r_spec, "RateQuoteExpression", "Percentage")) else 1.0)
        for tk, r_spec in (req_model.Model.MarketDataSet.RepoSpreadCurves or {}).items()
    }
    assert rt_res.RepoRho and oracle.repo_rho and set(rt_res.RepoRho.keys()) == set(oracle.repo_rho.keys()), (
        f"Deal {tid} RepoRho missing or mismatched: rt={rt_res.RepoRho!r}, oracle={oracle.repo_rho!r}"
    )
    rt_rho = to_decimal_merged_rho(rt_res, rate_scales, repo_quote_scales=repo_scales)
    or_rho = to_decimal_merged_rho(
        {"Rho": oracle.rho, "RepoRho": oracle.repo_rho, "rho_unit_convention": "quote"},
        rate_scales,
        repo_quote_scales=repo_scales,
    )
    assert len(or_rho) > 0 and set(rt_rho.keys()) == set(or_rho.keys())
    for cid, ov_vec in or_rho.items():
        assert abs(float(np.sum(rt_rho[cid])) - float(np.sum(ov_vec))) * 1e-3 * 1e4 < 5.0

    assert rt_res.DividendsSensitivities is not None and oracle.dividends_sensitivities is not None
    for tk, ov_div in oracle.dividends_sensitivities.items():
        assert abs(float(np.sum(rt_res.DividendsSensitivities[tk])) - float(np.sum(ov_div))) * 1e4 < 5.0

    assert rt_res.Theta is not None and oracle.theta is not None
    assert abs(float(rt_res.Theta) - float(oracle.theta)) * 1e4 < 5.0

    if or_cega:
        assert isinstance(rt_res.Cega, dict)
        for a, row_or_c in or_cega.items():
            row_rt_c = rt_res.Cega[a]
            assert isinstance(row_rt_c, dict)
            for b, ov in row_or_c.items():
                assert abs(float(row_rt_c[b]) - float(ov)) * 0.01 * 1e4 < 5.0


def test_audit_shard_regressions_empty_nan_f16_curves_api() -> None:
    import dataclasses

    from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCGpuSolver as CpuDupireSolver
    from deepvol.models.autocall.feature_layout import (
        AutocallGlobalCol,
        AutocallNodeCol,
        AutocallSchedCol,
    )
    from deepvol.models.autocall.greeks_engine import (
        _eval_autocall_payoff_pv_batch_np,
        compute_batched_surrogate_greeks,
    )
    from deepvol.mrm.eq_autocall_guardian import EQAutocallModelGuardian

    base_req = _load_base_request()
    req2 = _slice_assets(base_req, 2)
    st2 = preprocess_eq_autocall(req2)

    # Empty batch_size=0, m_max=0, d_max=0 guards + explicit lower_strike kwarg parity in batched surrogate Greeks.
    surr = MaskedEQAutocallSurrogate(d_max=8, m_max=48, d_model=64, n_egno_layers=2, n_sched_layers=1).eval()
    surr.zero_init_residual_heads(std=1e-3)
    for b_sz, d_dim, m_dim in ((0, 2, 4), (2, 2, 0), (2, 0, 4)):
        empty_res = compute_batched_surrogate_greeks(
            surr,
            torch.zeros((b_sz, d_dim, len(AutocallNodeCol)), dtype=torch.float64),
            torch.zeros((b_sz, d_dim, d_dim, 6), dtype=torch.float64),
            torch.ones((b_sz, d_dim), dtype=torch.float64),
            torch.zeros((b_sz, m_dim, len(AutocallSchedCol)), dtype=torch.float64),
            torch.ones((b_sz, m_dim), dtype=torch.float64),
            torch.zeros((b_sz, len(AutocallGlobalCol)), dtype=torch.float64),
        )
        assert empty_res.product_value.shape == (b_sz,)
        assert torch.all(torch.isfinite(empty_res.product_value))

    n_b = st2.node_feats.clone().to(torch.float64)
    e_b = st2.edge_feats.clone().to(torch.float64)
    am_b = st2.asset_mask.clone().to(torch.float64)
    sf_b = st2.sched_feats.clone().to(torch.float64)
    sm_b = st2.sched_mask.clone().to(torch.float64)
    gf_b = st2.global_feats.clone().to(torch.float64)
    gf_k = gf_b.clone()
    gf_k[:, int(AutocallGlobalCol.LOWER_STRIKE)] = 0.5
    out_kw = compute_batched_surrogate_greeks(
        surr, n_b, e_b, am_b, sf_b, sm_b, gf_b, lower_strike=0.5, forward_perf=st2.forward_perf.unsqueeze(0), v_perf=st2.v_perf.unsqueeze(0)
    )
    out_col = compute_batched_surrogate_greeks(
        surr, n_b, e_b, am_b, sf_b, sm_b, gf_k, forward_perf=st2.forward_perf.unsqueeze(0), v_perf=st2.v_perf.unsqueeze(0)
    )
    assert float(out_kw.put_loss.item()) == pytest.approx(float(out_col.put_loss.item()), rel=1e-9, abs=1e-12)

    # Inactive-slot NaN isolation and empty-asset safety in EQAutocallModelGuardian tensor batch gates.
    guardian = EQAutocallModelGuardian(mahalanobis_chi2_threshold=200.0, tau_desk_bps=50.0)
    synth = generate_synthetic_autocall_batch(batch_size=2, d_max=6, m_max=10, n_mc_paths=32, seed=31, device="cpu", dtype=torch.float64)
    n_nan = synth.node_feats.clone()
    s_nan = synth.sched_feats.clone()
    am_mask = synth.asset_mask.clone()
    sm_mask = synth.sched_mask.clone()
    am_mask[:, -1] = 0.0
    sm_mask[:, -1] = 0.0
    n_nan[:, -1, :] = float("nan")
    s_nan[:, -1, :] = float("nan")
    pre_nan = guardian.evaluate_tensor_batch_pre(
        n_nan, synth.edge_feats, am_mask, s_nan, sm_mask, synth.global_feats, record_drift=False
    )
    assert math.isfinite(float(pre_nan["max_conformal_half_width_bps"]))
    assert pre_nan["non_finite_count"] == 0
    post_nan = guardian.evaluate_tensor_batch_post(
        torch.full((2, 10), 0.05, dtype=torch.float64),
        torch.full((2, 10), 0.03, dtype=torch.float64),
        torch.full((2,), 0.02, dtype=torch.float64),
        torch.full((2,), 0.98, dtype=torch.float64),
        sm_mask,
        synth.global_feats,
        sched_feats=s_nan,
        node_feats=n_nan,
        asset_mask=am_mask,
    )
    assert math.isfinite(float(post_nan["residual_outlier_count"]))

    pre_d0 = guardian.evaluate_tensor_batch_pre(
        torch.zeros((2, 0, len(AutocallNodeCol)), dtype=torch.float64),
        torch.zeros((2, 0, 0, 6), dtype=torch.float64),
        torch.zeros((2, 0), dtype=torch.float64),
        synth.sched_feats,
        synth.sched_mask,
        synth.global_feats,
        record_drift=False,
    )
    assert pre_d0["batch_size"] == 2 and pre_d0["non_finite_count"] == 0

    # float16 MaskedEQAutocallSurrogate with q_proxy==0 must never divide 0/0 -> NaN.
    surr_f16 = MaskedEQAutocallSurrogate(d_max=4, m_max=8, d_model=32, n_egno_layers=1, n_sched_layers=1).eval().half()
    n16 = synth.node_feats[:, :4, :].to(torch.float16)
    e16 = synth.edge_feats[:, :4, :4, :].to(torch.float16)
    a16 = synth.asset_mask[:, :4].to(torch.float16)
    s16 = synth.sched_feats[:, :8, :].clone().to(torch.float16)
    s16[:, :, int(AutocallSchedCol.PROXY_CPN_PROB)] = 0.0
    sm16 = synth.sched_mask[:, :8].to(torch.float16)
    g16 = synth.global_feats.to(torch.float16)
    with torch.no_grad():
        out16 = surr_f16(n16, e16, a16, s16, sm16, g16)
    assert (
        torch.isfinite(out16.stopping_probs).all()
        and torch.isfinite(out16.coupon_probs).all()
        and torch.isfinite(out16.put_loss).all()
    )

    # Batched B=2 OneStar distinct correlation rows + d=1 OneStar ko==ki non-double-counting parity.
    c0 = torch.eye(3, dtype=torch.float64) * 0.9 + 0.1
    c1 = torch.eye(3, dtype=torch.float64) * 0.2 + 0.8
    st_os_b2 = {
        "forward_perf": torch.tensor([[[1.02, 1.01], [0.98, 0.99], [1.05, 1.04]], [[1.02, 1.01], [0.98, 0.99], [1.05, 1.04]]], dtype=torch.float64),
        "v_perf": torch.tensor([[[0.22, 0.28], [0.24, 0.30], [0.20, 0.26]], [[0.22, 0.28], [0.24, 0.30], [0.20, 0.26]]], dtype=torch.float64),
        "corr_matrix": torch.stack([c0, c1], dim=0),
        "df_pay": torch.tensor([[0.96, 0.92], [0.96, 0.92]], dtype=torch.float64),
        "ko_barriers": torch.tensor([[1.10, 1.05], [1.10, 1.05]], dtype=torch.float64),
        "is_callable": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        "cpn_barriers": torch.tensor([[0.80, 0.80], [0.80, 0.80]], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        "coupon_amounts": torch.tensor([[0.04, 0.04], [0.04, 0.04]], dtype=torch.float64),
        "asset_mask": torch.ones((2, 3), dtype=torch.float64),
        "sched_mask": torch.ones((2, 2), dtype=torch.float64),
        "one_star_strike": 1.02,
        "one_star_coupon": 0.06,
    }
    res_b2 = ab.compute_analytical_autocall_baseline(st_os_b2, use_fast_cpu=False)
    st_os_1 = {
        k: (v[1] if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
        for k, v in st_os_b2.items()
        if k not in ("asset_mask", "sched_mask")
    }
    res_single1 = ab.compute_analytical_autocall_baseline(st_os_1, use_fast_cpu=False)
    assert float(res_b2.product_value_proxy[1].item()) == pytest.approx(float(res_single1.product_value_proxy.item()), abs=1e-9)

    st_koki = {
        "forward_perf": torch.tensor([[1.0, 0.95]], dtype=torch.float64),
        "v_perf": torch.tensor([[0.25, 0.35]], dtype=torch.float64),
        "corr_matrix": torch.tensor([[1.0]], dtype=torch.float64),
        "df_pay": torch.tensor([0.96, 0.92], dtype=torch.float64),
        "ko_barriers": torch.tensor([1.10, 0.75], dtype=torch.float64),
        "is_callable": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "cpn_barriers": torch.tensor([0.75, 0.75], dtype=torch.float64),
        "is_guaranteed_cpn": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "coupon_amounts": torch.tensor([0.03, 0.03], dtype=torch.float64),
        "strike": 0.90,
        "ki_barrier": 0.75,
        "has_ki": True,
        "i_ki_past": 0.0,
        "one_star_strike": 1.02,
        "one_star_coupon": 0.05,
        "allow_autocall_on_expiry": True,
    }
    pv_koki_torch = ab.compute_analytical_autocall_baseline(st_koki, use_fast_cpu=False)
    pv_koki_cpp = ab.compute_analytical_autocall_baseline(st_koki, use_fast_cpu=True)
    assert float(pv_koki_torch.product_value_proxy.item()) == pytest.approx(
        float(pv_koki_cpp.product_value_proxy.item()), abs=1e-8
    )

    # MonotoneConvex tau=0 with negative start tenor, Thirty360 funding + payment_dates=None, TwinWin d=1, already_autocalled.
    val_dt = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    c_neg_t = ClampedYieldCurve(
        as_of_date=val_dt,
        pillar_tenors=np.array([-0.25, 0.25, 1.0], dtype=np.float64),
        rates=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        interpolation_method="MonotoneConvex",
    )
    dfs_z, jac_z = c_neg_t.df_and_d_df_d_pillar([0.0, 0.5])
    assert np.all(np.isfinite(dfs_z)) and np.all(np.isfinite(jac_z))

    cs_none_pay = collapse_seasoned_schedule(
        as_of_date=datetime.datetime(2025, 1, 15, tzinfo=datetime.timezone.utc),
        start_date=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        observation_dates=[datetime.datetime(2025, 1, 31, tzinfo=datetime.timezone.utc), datetime.datetime(2025, 3, 31, tzinfo=datetime.timezone.utc)],
        payment_dates=[datetime.datetime(2025, 2, 28, tzinfo=datetime.timezone.utc), datetime.datetime(2025, 3, 31, tzinfo=datetime.timezone.utc)],
        underlying_identifiers=["A"],
        underlying_reference_spots=[100.0],
        spots_t0=[100.0],
        ko_barriers=[1.10, 1.10],
        cpn_barriers=[0.80, 0.80],
        fixed_coupon_amount=0.04,
        funding_leg={"funding_spread": 0.01, "funding_day_count": "Thirty360", "payment_dates": None},
    )
    assert cs_none_pay.already_autocalled is False

    req1 = _slice_assets(base_req, 1)
    st1_tw = preprocess_eq_autocall(copy.deepcopy(req1))
    st1_tw = dataclasses.replace(st1_tw, option_index="TwinWin", state_digest="tw_d1_test")
    perf_tw = np.full((1, 1, max(st1_tw.m_rem, 1), 4), 0.70, dtype=np.float64)
    pv_tw = _eval_autocall_payoff_pv_batch_np(perf_tw, st1_tw)
    st1_wof = dataclasses.replace(st1_tw, option_index="WorstOf", state_digest="wof_d1_test")
    pv_wof = _eval_autocall_payoff_pv_batch_np(perf_tw, st1_wof)
    assert float(pv_tw[0]) != float(pv_wof[0])

    st_auto = dataclasses.replace(st2, already_autocalled=True)
    res_auto = CpuDupireSolver(device="cpu").simulate_and_price(st_auto)
    assert math.isfinite(res_auto.product_value) and abs(res_auto.option_pv) <= 1e-12

    # API runtime: no_grad CPU microbatch Greeks, price_batch_json NaN vs None + ndarray, malformed shapes HTTP 422.
    rt_cpu = EQAutocallRuntime(device="cpu", enable_compiled_runner=False)
    with torch.no_grad():
        tb_ng = rt_cpu.price_tensor_batch(
            synth.node_feats, synth.edge_feats, synth.asset_mask, synth.sched_feats, synth.sched_mask, synth.global_feats, compute_greeks=True
        )
    assert tb_ng.product_value.shape == (2,) and torch.all(torch.isfinite(tb_ng.product_value))

    good_d = req2.model_dump(mode="json")
    good_d.pop("stripped_legacy_fields", None)
    good_d["Model"]["MarketDataSet"]["RatesCurvesBundles"]["RUB_RUONIA_OIS"]["RatesSchedule"]["Spreads"] = None
    nan_d = copy.deepcopy(good_d)
    nan_d["Model"]["MarketDataSet"]["RatesCurvesBundles"]["RUB_RUONIA_OIS"]["RatesSchedule"]["Spreads"] = [float("nan")] * len(
        nan_d["Model"]["MarketDataSet"]["RatesCurvesBundles"]["RUB_RUONIA_OIS"]["RatesSchedule"]["Dates"]
    )
    with pytest.raises(ValidationError):
        rt_cpu.price_batch_json([good_d, nan_d], compute_greeks=False)

    arr_d = copy.deepcopy(good_d)
    arr_d["Product"]["AutoCallTriggerLevels"] = np.asarray(arr_d["Product"]["AutoCallTriggerLevels"], dtype=np.float64)
    arr_d["TradeInfo"] = ""
    arr_none_ti = copy.deepcopy(arr_d)
    arr_none_ti["TradeInfo"] = None
    out_arr = rt_cpu.price_batch_json([arr_d, arr_none_ti], compute_greeks=False)
    assert len(out_arr) == 2 and out_arr[0].TradeInfo == "" and out_arr[1].TradeInfo is None

    with TestClient(app) as cli:
        bad_sched = {
            "node_feats": synth.node_feats.tolist(),
            "edge_feats": synth.edge_feats.tolist(),
            "asset_mask": synth.asset_mask.tolist(),
            "sched_feats": synth.sched_feats[:, :, :7].tolist(),
            "sched_mask": synth.sched_mask.tolist(),
            "global_feats": synth.global_feats.tolist(),
        }
        assert cli.post("/eq_autocall/price_batch?compute_greeks=true", json=bad_sched).status_code == 422
        bad_edge = {**bad_sched, "sched_feats": synth.sched_feats.tolist(), "edge_feats": synth.edge_feats[:, :, :, :2].tolist()}
        assert cli.post("/eq_autocall/price_batch?compute_greeks=true", json=bad_edge).status_code == 422
        bad_bsz = {**bad_sched, "sched_feats": synth.sched_feats[:1].tolist()}
        assert cli.post("/eq_autocall/price_batch", json=bad_bsz).status_code == 422
        bad_m0 = {**bad_sched, "sched_feats": np.zeros((2, 0, len(AutocallSchedCol))).tolist(), "sched_mask": np.zeros((2, 0)).tolist()}
        assert cli.post("/eq_autocall/price_batch?compute_greeks=true", json=bad_m0).status_code == 422


def _assert_vram_under_250mb(
    base_res: int,
    max_res: int,
    *,
    label: str,
    subprocess_code: str,
) -> None:
    """Assert <= 250 MB peak reserved VRAM directly, or via delta + clean subprocess if mempool base_res > 64 MB."""
    if base_res > 64 * 1024 * 1024 and os.environ.get("_DEEPVOL_VRAM_SUBPROC") != "1":
        delta_mb = (max_res - base_res) / (1024.0 * 1024.0)
        assert delta_mb <= 250.0, (
            f"{label} delta VRAM {delta_mb:.1f} MB > 250.0 MB (base={base_res / (1024.0 * 1024.0):.1f} MB)"
        )
        sub_env = os.environ.copy()
        sub_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        sub_env["_DEEPVOL_VRAM_SUBPROC"] = "1"
        sub_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), str(REPO_ROOT / "src")]
            + ([sub_env["PYTHONPATH"]] if sub_env.get("PYTHONPATH") else [])
        )
        subprocess.run([sys.executable, "-c", textwrap.dedent(subprocess_code)], env=sub_env, check=True, timeout=120)
    else:
        res_mb = max_res / (1024.0 * 1024.0)
        assert res_mb <= 250.0, f"{label} peak reserved VRAM {res_mb:.1f} MB > 250.0 MB"


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_dual_gate_latency_and_vram_audit() -> None:
    import gc
    import re

    from deepvol.api.eq_autocall_runtime import EQAutocallRuntimeResponse, _attach_runtime_market_metadata
    from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER
    from deepvol.models.autocall.qmc_aad_greeks import QmcAadGreeksConfig, compute_qmc_aad_crn_greeks

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for Dual-Gate GPU SLA verification")

    ps_root = REPO_ROOT / "ps_audit_b10316f2"
    manifest_path = ps_root / "deals" / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("ps_audit_b10316f2/deals/manifest.json not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = EQAutocallRuntime(device="cuda")
    solver = DupireQMCGpuSolver(device="cuda", backend="triton")
    cfg_65k = QmcAadGreeksConfig(n_paths=65536, n_replicates=8, n_substeps=4, lv_grid_nt=32, lv_grid_ny=64, seed=42)
    cfg_mf = DupireQMCConfig(n_paths=65536, n_replicates=8, n_substeps=4, cv_mode="multifidelity_gbm")
    oracle_queries = ["ProductValue", "CashFlows"] + list(ALL_8_GREEK_QUERIES)

    GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_res = torch.cuda.memory_reserved()
    warm_ms: list[float] = []
    all_sweep_ms: list[float] = []
    priced_items: list[tuple[str, EQAutocallRuntimeResponse, EQAutocallableTradeRequest]] = []

    docs_req_dir = REPO_ROOT / "docs" / "eq_autocall_ps_batch_b10316f2" / "requests"
    for deal in manifest["deals"]:
        tid = str(deal["trade_id"])
        canonical_req = docs_req_dir / str(deal["filename"])
        req_path = (
            canonical_req
            if tid == "56653809" and canonical_req.is_file()
            else ps_root / str(deal["paths"]["full_ps_request_drop_target"])
        )
        exp_path = ps_root / str(deal["paths"]["expected_murex"])
        assert req_path.is_file() and exp_path.is_file(), f"Missing PS files for deal {tid}"
        req_dict = json.loads(req_path.read_text(encoding="utf-8"))
        exp_dict = json.loads(exp_path.read_text(encoding="utf-8"))
        if deal["status"] == "rejected":
            with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
                runtime.price_trade(req_dict, engine="auto", include_all_greeks=True)
            t0 = time.perf_counter()
            with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
                runtime.price_trade(req_dict, engine="auto", include_all_greeks=True)
            if sys.gettrace() is None:
                assert (time.perf_counter() - t0) * 1000.0 < 1.0
            continue

        req_dict["Queries"] = list(oracle_queries)
        _ = runtime.price_trade(req_dict, engine="auto", include_all_greeks=True)
        _ = runtime.price_trade(req_dict, engine="auto", include_all_greeks=True)
        gc.collect()
        torch.cuda.synchronize()
        deal_ts: list[float] = []
        rt_res = None
        for _ in range(3):
            t0 = time.perf_counter()
            rt_res = runtime.price_trade(req_dict, engine="auto", include_all_greeks=True)
            torch.cuda.synchronize()
            deal_ts.append((time.perf_counter() - t0) * 1000.0)
        assert rt_res is not None
        warm_ms.append(float(np.median(deal_ts)))
        all_sweep_ms.extend(deal_ts)
        if sys.gettrace() is None:
            assert max(deal_ts) < 10.0, f"Deal {tid} sweep latency {max(deal_ts):.2f} ms >= 10.0 ms ({deal_ts})"

        murex_pv = float(exp_dict["ProductValue"])
        murex_tol_bps = max(5.0, float(exp_dict.get("StandardError", 0.0)) * 1e4)
        assert abs(rt_res.ProductValue - murex_pv) * 1e4 < murex_tol_bps
        req_m = EQAutocallableTradeRequest.model_validate(req_dict)
        priced_items.append((tid, rt_res, req_m))

    assert REF_REQUEST_PATH.is_file(), f"Missing reference DAG at {REF_REQUEST_PATH}"
    dag_text = REF_REQUEST_PATH.read_text(encoding="utf-8")
    m_ref = re.search(r"(?m)^@[0-9a-fA-F]{40}\s*$", dag_text)
    header = dag_text[: m_ref.start()] if m_ref else dag_text
    pv_m = re.search(r"ProductValue:\s*\n\s*([-+0-9.eE]+)", header)
    assert pv_m is not None
    mx_45_pv = float(pv_m.group(1))
    cf_45_mx: dict[tuple[str, str], float] = {}
    cf_blk = re.search(r"CashFlows:\s*\n\s*\[(.*?)\n\s*\]", header, flags=re.DOTALL)
    if cf_blk is not None:
        for obj_m in re.finditer(r"\{(.*?)\}", cf_blk.group(1), flags=re.DOTALL):
            blk = obj_m.group(1)
            dm = re.search(r"Description:\s*\n\s*'([^']+)'", blk)
            dtm = re.search(r"Date:\s*\n\s*datetime\.datetime\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", blk)
            am = re.search(r"Amount:\s*\n\s*([-+0-9.eE]+)", blk)
            if dm and dtm and am:
                k = (f"{int(dtm.group(1)):04d}-{int(dtm.group(2)):02d}-{int(dtm.group(3)):02d}", dm.group(1))
                cf_45_mx[k] = cf_45_mx.get(k, 0.0) + float(am.group(1))

    _ = runtime.price_murex_dag(dag_text, include_all_greeks=True)
    _ = runtime.price_murex_dag(dag_text, include_all_greeks=True)
    torch.cuda.synchronize()
    deal_45_ts: list[float] = []
    rt_45 = None
    for _ in range(3):
        t0 = time.perf_counter()
        rt_45 = runtime.price_murex_dag(dag_text, include_all_greeks=True)
        torch.cuda.synchronize()
        deal_45_ts.append((time.perf_counter() - t0) * 1000.0)
    assert rt_45 is not None
    warm_ms.append(float(np.median(deal_45_ts)))
    all_sweep_ms.extend(deal_45_ts)
    if sys.gettrace() is None:
        assert max(deal_45_ts) < 10.0, f"Deal 45476647 sweep latency {max(deal_45_ts):.2f} ms >= 10.0 ms ({deal_45_ts})"
    assert abs(rt_45.ProductValue - mx_45_pv) * 1e4 < 5.0
    cf_45_rt = _audit_group_cashflows(rt_45.CashFlows)
    for k in sorted(set(cf_45_rt) | set(cf_45_mx)):
        assert abs(cf_45_rt.get(k, 0.0) - cf_45_mx.get(k, 0.0)) * 1e4 < 5.0
    req_45, _ = parse_murex_autocall_dag(dag_text)
    priced_items.append(("45476647", rt_45, req_45))

    torch.cuda.synchronize()
    _assert_vram_under_250mb(
        base_res,
        torch.cuda.max_memory_reserved(),
        label="Runtime sweep",
        subprocess_code=f"""
            import gc, json, torch
            from pathlib import Path
            from deepvol.api.eq_autocall_runtime import ALL_8_GREEK_QUERIES, EQAutocallRuntime
            from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER
            ps_root = Path({str(ps_root)!r})
            docs_req_dir = Path({str(docs_req_dir)!r})
            manifest = json.loads((ps_root / "deals" / "manifest.json").read_text(encoding="utf-8"))
            rt = EQAutocallRuntime(device="cuda")
            qs = ["ProductValue", "CashFlows"] + list(ALL_8_GREEK_QUERIES)
            GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for deal in manifest["deals"]:
                if deal["status"] == "rejected":
                    continue
                canonical_req = docs_req_dir / str(deal["filename"])
                req_p = (
                    canonical_req
                    if str(deal["trade_id"]) == "56653809" and canonical_req.is_file()
                    else ps_root / str(deal["paths"]["full_ps_request_drop_target"])
                )
                req_d = json.loads(req_p.read_text(encoding="utf-8"))
                req_d["Queries"] = list(qs)
                rt.price_trade(req_d, engine="auto", include_all_greeks=True)
            dag_p = Path({str(REF_REQUEST_PATH)!r})
            assert dag_p.is_file()
            rt.price_murex_dag(dag_p.read_text(encoding="utf-8"), include_all_greeks=True)
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_reserved() <= 250.0 * 1024 * 1024
        """,
    )
    assert len(priced_items) == 13
    if sys.gettrace() is None:
        assert float(np.median(warm_ms)) <= 7.0
        assert max(all_sweep_ms) < 10.0

    for tid, rt_res, req_m in priced_items:
        st = preprocess_eq_autocall(req_m, d_max=12, m_max=64)
        _attach_runtime_market_metadata(st, req_m)
        oracle = compute_qmc_aad_crn_greeks(
            st, request=req_m, queries=oracle_queries, config=cfg_65k, device=torch.device("cuda")
        )
        mf_res = solver.simulate_and_price(st, config=cfg_mf)
        _audit_assert_dual_gate_oracle(tid, rt_res, oracle, float(mf_res.product_value), st, req_m)

    # Also verify d=12, M=60 multifidelity_gbm standalone peak VRAM <= 250 MB.
    st_max = preprocess_eq_autocall(_build_dim_horizon_request(_load_base_request(), d_assets=12, m_periods=60), d_max=12, m_max=60)
    GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_res_max = torch.cuda.memory_reserved()
    res_max = solver.simulate_and_price(
        st_max, DupireQMCConfig(n_paths=8192, n_replicates=8, n_substeps=4, cv_mode="multifidelity_gbm")
    )
    assert np.isfinite(res_max.product_value)
    torch.cuda.synchronize()
    _assert_vram_under_250mb(
        base_res_max,
        torch.cuda.max_memory_reserved(),
        label="M=60 d=12 multifidelity_gbm",
        subprocess_code=f"""
            import gc, sys, torch
            from pathlib import Path
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig, DupireQMCGpuSolver
            from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER
            from deepvol.models.autocall.pre_post_processor import preprocess_eq_autocall
            from test_eq_autocall_defensive_audit import _build_dim_horizon_request, _load_base_request
            solver = DupireQMCGpuSolver(device="cuda", backend="triton")
            st_max = preprocess_eq_autocall(_build_dim_horizon_request(_load_base_request(), d_assets=12, m_periods=60), d_max=12, m_max=60)
            GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            solver.simulate_and_price(st_max, DupireQMCConfig(n_paths=8192, n_replicates=8, n_substeps=4, cv_mode="multifidelity_gbm"))
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_reserved() <= 250.0 * 1024 * 1024
        """,
    )
