"""Permanent 131-scenario adversarial matrix covering SC-001..SC-131 across M1-M6."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import datetime
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError
import pytest
import torch

from deepvol.api.schemas_eq_autocall import EQAutocallableTradeRequest, parse_murex_autocall_dag
from deepvol.models.autocall import (
    AutocallPreprocState,
    assemble_autocall_cashflows_and_pv,
    compute_analytical_autocall_baseline,
    pre_post_processor as ppp_mod,
    preprocess_eq_autocall,
    price_eq_autocall_dupire_qmc,
)

_LOCAL_DATA_REQUEST = Path(__file__).resolve().parents[1] / "data" / "autocall" / "eq_autocallable_request.txt"
_MAIN_REPO_REQUEST = Path("/home/execorn/programming/derivatives/data/autocall/eq_autocallable_request.txt")
REQUEST_TXT_PATH = _LOCAL_DATA_REQUEST if _LOCAL_DATA_REQUEST.exists() else _MAIN_REPO_REQUEST

_MONITORING_ALL = (
    ("SC-069", "WorstOf"),
    ("SC-070", "BestOf"),
    ("SC-071", "Average"),
    ("SC-072", "AverageOf"),
    ("SC-073", "Outperformance"),
    ("SC-074", "TwinWin"),
    ("SC-075", "Rainbow"),
    ("SC-076", "Lookback"),
    ("SC-077", "Spread"),
    ("SC-078", "Dispersion"),
    ("SC-079", "RatioAB"),
    ("SC-080", "RatioBA"),
    ("SC-081", "RatioBestWorst"),
    ("SC-082", "RatioBestAverage"),
    ("SC-083", "RatioAverageWorst"),
    ("SC-084", "DiffAB"),
    ("SC-085", "DiffBA"),
    ("SC-086", "DiffBestWorst"),
    ("SC-087", "DiffBestAverage"),
    ("SC-088", "DiffAverageWorst"),
    ("SC-132", "EquallyWeightedBasket"),
    ("SC-133", "SecondWorstOf"),
    ("SC-134", "ThirdWorstOf"),
    ("SC-135", "SecondBestOf"),
    ("SC-136", "MultiPerf"),
)
_ALL_8_GREEK_QUERIES = [
    "Delta",
    "Gamma",
    "MultivariateAggregatedCashGamma",
    "Vega",
    "Rho",
    "DividendsSensitivities",
    "Theta",
    "Cega",
]


def utc_dt(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime.datetime:
    return datetime.datetime(y, m, d, hh, mm, tzinfo=datetime.timezone.utc)


def _make_past_fixings(
    tickers: list[str],
    dates: list[datetime.datetime],
    vals_per_asset: list[list[float]],
) -> dict[str, Any]:
    return {
        tk: {
            "TypeName": "Fixings",
            "Identifier": tk,
            "Dates": list(dates),
            "Values": list(vals_per_asset[i]),
        }
        for i, tk in enumerate(tickers)
    }


def make_synthetic_request_dict(
    *,
    d_assets: int = 3,
    m_periods: int = 4,
    as_of: Optional[datetime.datetime] = None,
    start_date: Optional[datetime.datetime] = None,
    obs_dates: Optional[list[datetime.datetime]] = None,
    pay_dates: Optional[list[datetime.datetime]] = None,
    spots: Optional[list[float]] = None,
    ref_spots: Optional[list[float]] = None,
    vol_level: float = 0.30,
    vol_grid_override: Optional[list[list[float]]] = None,
    strikes_mult: Optional[list[float]] = None,
    exp_days: Optional[list[int]] = None,
    corr_val: float = 0.45,
    corr_matrix: Optional[npt.NDArray[np.float64]] = None,
    rate_pct: float = 10.0,
    rate_curve_pcts: Optional[list[float]] = None,
    repo_pct: float = 1.0,
    mixed_repos: Optional[list[Optional[float]]] = None,
    include_repo: bool = True,
    div_events_per_asset: Optional[dict[int, list[tuple[datetime.datetime, datetime.datetime, float]]]] = None,
    ko_levels: Optional[list[float]] = None,
    cpn_levels: Optional[list[float]] = None,
    fixed_cpn_amt: Union[float, list[float], None] = 0.04,
    fixed_rate: Union[float, list[float], None] = None,
    is_memory: bool = True,
    strike: float = 0.90,
    lower_strike: Optional[float] = None,
    option_gearing: float = 1.0,
    ki_barrier: Optional[float] = None,
    ki_obs_type: str = "Discrete",
    ko_spread: float = -0.15,
    cpn_spread: Optional[float] = None,
    ki_spread: Optional[float] = None,
    callability_index: str = "WorstOf",
    coupon_index: Optional[str] = None,
    option_index: Optional[str] = None,
    coupon_barrier_type: str = "In",
    coupon_barrier_direction: str = "Up",
    no_notional_repayment: bool = False,
    autocall_coupon: float = 0.0,
    force_coupons_on_autocall: bool = False,
    allow_autocall_on_expiry: bool = False,
    is_long_basket: bool = False,
    part_call_strike: Optional[float] = None,
    part_call_rate: Optional[float] = None,
    one_star_strike: Optional[float] = None,
    one_star_coupon: Optional[float] = None,
    rate_identifier: Optional[str] = None,
    funding_spread: Optional[float] = None,
    funding_leg_dates: Optional[list[datetime.datetime]] = None,
    predefined_divs: Optional[dict[str, dict[str, Any]]] = None,
    fixings_override: Optional[dict[str, dict[str, Any]]] = None,
    asset_ccy_suffix: str = "RUB",
    payment_ccy: str = "RUB",
    fx_vol: Optional[float] = None,
    fx_corr: Optional[float] = None,
    dcc: str = "Act360",
    tickers_override: Optional[list[str]] = None,
) -> dict[str, Any]:
    t0 = as_of or utc_dt(2026, 1, 15)
    t_start = start_date or (t0 - datetime.timedelta(days=30))
    if obs_dates is None:
        obs_dates = [t0 + datetime.timedelta(days=90 * (k + 1)) for k in range(m_periods)]
    if pay_dates is None:
        pay_dates = [d + datetime.timedelta(days=3) for d in obs_dates]
    m = len(obs_dates)

    tickers = list(tickers_override) if tickers_override is not None else [f"EQ{i:02d}_{asset_ccy_suffix}" for i in range(d_assets)]
    d = len(tickers)
    ref_spots = [100.0] * d if ref_spots is None else list(ref_spots)
    spots = [100.0] * d if spots is None else list(spots)
    ko_levels = ([1.00] * (m - 1) + [500.0] if m > 0 else []) if ko_levels is None else list(ko_levels)
    cpn_levels = ([0.75] * m) if cpn_levels is None else list(cpn_levels)

    if exp_days is None:
        max_days = max(int((obs_dates[-1] - t0).days) + 30, 365) if m > 0 else 365
        exp_days = sorted({30, 90, 180, 365, max(540, max_days), max(730, max_days + 180)})
    exp_dts = [t0 + datetime.timedelta(days=dd) for dd in exp_days]
    strikes_mult = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5] if strikes_mult is None else list(strikes_mult)

    vol_surfaces: dict[str, Any] = {}
    for i, tk in enumerate(tickers):
        s_base = max(abs(ref_spots[i]), 1.0)
        k_list = [s_base * sm for sm in strikes_mult]
        vq = copy.deepcopy(vol_grid_override) if vol_grid_override is not None else [[float(vol_level) for _ in k_list] for _ in exp_dts]
        vol_surfaces[tk] = {
            "Identifier": tk,
            "TypeName": "EquityVolatilitySmileCollection",
            "DayCountConvention": "Act365",
            "SmileInterpolationMethod": "CubicSpline",
            "SmileExtrapolationMethod": "Linear",
            "ExpirationDates": exp_dts,
            "Strikes": k_list,
            "VolatilityQuotes": vq,
        }

    fx_id = None
    if fx_vol is not None and asset_ccy_suffix != payment_ccy:
        fx_id = f"{asset_ccy_suffix}_{payment_ccy}"
        vol_surfaces[fx_id] = {
            "Identifier": fx_id,
            "TypeName": "FXVolatilitySmileCollection",
            "AsOfDate": t0,
            "DayCountConvention": "Act365",
            "SmileInterpolationMethod": "CubicSpline",
            "SmileExtrapolationMethod": "Flat",
            "Deltas": ["25P", "ATM", "25C"],
            "Tenors": ["30D", "180D", "365D", "1095D"],
            "VolatilityQuotes": [[fx_vol, fx_vol, fx_vol] for _ in range(4)],
        }

    corrs: dict[str, float] = {}
    for i in range(d):
        for j in range(i + 1, d):
            corrs[f"{tickers[i]}&{tickers[j]}"] = float(corr_matrix[i, j]) if corr_matrix is not None else float(corr_val)
    if fx_id is not None and fx_corr is not None:
        for tk in tickers:
            corrs[f"{tk}&{fx_id}"] = float(fx_corr)

    curve_days = [30, 90, 180, 365, 730, 1460, 2190, 3650]
    curve_dates = [t0 + datetime.timedelta(days=dd) for dd in curve_days]
    if rate_curve_pcts is not None:
        r_vals = list(rate_curve_pcts)
        curve_dates = curve_dates[: len(r_vals)]
    else:
        r_vals = [float(rate_pct) for _ in curve_dates]

    rates_bundles = {
        f"{payment_ccy}_OIS": {
            "TypeName": "DiscountFactorCurve",
            "Identifier": f"{payment_ccy}_OIS",
            "Currency": payment_ccy,
            "AsOfDate": t0,
            "InterpolationDayCountConvention": "Act365",
            "InterpolationMethod": "CubicSpline",
            "InterpolationSpaceCoordinates": "Rates",
            "ExtrapolationMethod": "Flat",
            "RateQuoteExpression": "Percentage",
            "RatesSchedule": {"Dates": curve_dates, "Rates": r_vals},
        }
    }

    repo_curves: dict[str, Any] = {}
    if mixed_repos is not None:
        for tk, rp in zip(tickers, mixed_repos):
            if rp is not None:
                repo_curves[tk] = {
                    "TypeName": "RepoSpreadCurve",
                    "Identifier": tk,
                    "InterpolationDayCountConvention": "Act360",
                    "InterpolationMethod": "CubicSpline",
                    "ExtrapolationMethod": "Flat",
                    "InterpolationSpaceCoordinates": "Rates",
                    "RateQuoteExpression": "Percentage",
                    "RatesSchedule": {"Periods": ["30D", "180D", "365D", "1095D"], "Rates": [float(rp)] * 4},
                }
    elif include_repo:
        for tk in tickers:
            repo_curves[tk] = {
                "TypeName": "RepoSpreadCurve",
                "Identifier": tk,
                "InterpolationDayCountConvention": "Act360",
                "InterpolationMethod": "CubicSpline",
                "ExtrapolationMethod": "Flat",
                "InterpolationSpaceCoordinates": "Rates",
                "RateQuoteExpression": "Percentage",
                "RatesSchedule": {"Periods": ["30D", "180D", "365D", "1095D"], "Rates": [float(repo_pct)] * 4},
            }

    dividends: dict[str, Any] = {}
    for i, tk in enumerate(tickers):
        ev_list = []
        if div_events_per_asset and i in div_events_per_asset:
            for ex_d, pay_d, amt in div_events_per_asset[i]:
                ev_list.append({"ExDate": ex_d, "PaymentDate": pay_d, "Amount": float(amt)})
        dividends[tk] = {"TypeName": "Dividends", "Identifier": tk, "CashDividendList": ev_list}

    product_dict: dict[str, Any] = {
        "TypeName": "AutoCallableProduct",
        "PaymentCurrency": payment_ccy,
        "StartDate": t_start,
        "PaymentDates": pay_dates,
        "ObservationDates": obs_dates,
        "DayCountConvention": dcc,
        "AutoCallTriggerAbsoluteDigitalSpread": float(ko_spread),
        "Strike": float(strike),
        "OptionGearing": float(option_gearing),
        "CallabilityMonitoringIndex": callability_index,
        "AutoCallTriggerLevels": [float(x) for x in ko_levels],
        "UnderlyingIdentifiers": tickers,
        "UnderlyingReferenceSpots": [float(x) for x in ref_spots],
        "IsMemory": bool(is_memory),
        "CouponBarrierLevels": [float(x) for x in cpn_levels],
        "CouponBarrierType": coupon_barrier_type,
        "CouponBarrierDirection": coupon_barrier_direction,
        "NoNotionalRepayment": bool(no_notional_repayment),
        "ForceCouponsOnAutocall": bool(force_coupons_on_autocall),
        "AllowAutocallOnExpiry": bool(allow_autocall_on_expiry),
        "IsLongBasket": bool(is_long_basket),
    }
    if fixed_rate is not None:
        product_dict["FixedRate"] = fixed_rate
    elif fixed_cpn_amt is not None:
        product_dict["FixedCouponAmount"] = fixed_cpn_amt
    if lower_strike is not None:
        product_dict["LowerStrike"] = float(lower_strike)
    if ki_barrier is not None:
        product_dict["OptionDownAndInBarrierLevel"] = float(ki_barrier)
        product_dict["OptionDownAndInBarrierObservationType"] = ki_obs_type
    if cpn_spread is not None:
        product_dict["CouponBarrierAbsoluteDigitalSpread"] = float(cpn_spread)
    if ki_spread is not None:
        product_dict["OptionDownAndInBarrierAbsoluteDigitalSpread"] = float(ki_spread)
    if coupon_index is not None:
        product_dict["CouponBarrierMonitoringIndex"] = coupon_index
    if option_index is not None:
        product_dict["OptionMonitoringIndex"] = option_index
    if autocall_coupon != 0.0:
        product_dict["AutoCallCoupon"] = float(autocall_coupon)
    if part_call_strike is not None:
        product_dict["ParticipationCallStrike"] = float(part_call_strike)
    if part_call_rate is not None:
        product_dict["ParticipationCallRate"] = float(part_call_rate)
    if one_star_strike is not None:
        product_dict["OneStarStrike"] = float(one_star_strike)
    if one_star_coupon is not None:
        product_dict["OneStarCoupon"] = float(one_star_coupon)
    if rate_identifier is not None:
        product_dict["RateIdentifier"] = rate_identifier
    if funding_spread is not None:
        product_dict["FundingSpread"] = float(funding_spread)
    if funding_leg_dates is not None:
        product_dict["FundingLegPaymentDates"] = funding_leg_dates
    if predefined_divs is not None:
        product_dict["PredefinedDividends"] = predefined_divs

    return {
        "Product": product_dict,
        "Model": {
            "TypeName": "DupireLocalVolatilityModel",
            "FinancingCurveIdentifiers": [f"{payment_ccy}_OIS"],
            "MarketDataSet": {
                "TypeName": "MarketDataSet",
                "AsOfDate": t0,
                "RatesCurvesBundles": rates_bundles,
                "VolatilitySurfaces": vol_surfaces,
                "Spots": {tk: float(spots[i]) for i, tk in enumerate(tickers)},
                "Correlations": corrs,
                "RepoSpreadCurves": repo_curves,
                "Dividends": dividends,
                "Fixings": fixings_override or {},
            },
        },
        "Queries": ["ProductValue", "CashFlows"] + list(_ALL_8_GREEK_QUERIES),
        "TradeInfo": "SYNTHETIC_EXTREME_TEST",
    }


def _assert_all_finite(obj: Any, path: str = "root") -> None:
    if obj is None or isinstance(obj, (str, bool, datetime.datetime, datetime.date)):
        return
    if isinstance(obj, (int, float, np.floating, np.integer)):
        assert math.isfinite(float(obj)), f"Non-finite scalar at {path}: {obj}"
        return
    if isinstance(obj, np.ndarray):
        assert bool(np.all(np.isfinite(obj))), f"Non-finite ndarray at {path}"
        return
    if isinstance(obj, torch.Tensor):
        assert bool(torch.isfinite(obj).all().item()), f"Non-finite tensor at {path}"
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            _assert_all_finite(v, f"{path}[{idx}]")


def _assert_prob_and_put_bounds(
    *,
    p_stop: npt.NDArray[np.float64],
    q_cpn: npt.NDArray[np.float64],
    l_put: float,
    option_pv: float,
    state: AutocallPreprocState,
) -> None:
    assert np.all(np.isfinite(p_stop)) and np.all(np.isfinite(q_cpn))
    assert math.isfinite(l_put) and math.isfinite(option_pv)
    if len(p_stop) > 0:
        assert float(np.min(p_stop)) >= -1e-7
        assert float(np.max(p_stop)) <= 1.0 + 1e-7
        assert float(np.sum(p_stop)) <= 1.0 + 1e-6
    if len(q_cpn) > 0:
        assert float(np.min(q_cpn)) >= -1e-7
        assert float(np.max(q_cpn)) <= 1.0 + 1e-7

    surv_term = max(1.0 - float(np.sum(p_stop)), 0.0) if len(p_stop) > 0 else 0.0
    k_eff = max(float(state.strike), 0.0)
    if state.lower_strike is not None and state.lower_strike >= 0.0:
        k_eff = max(k_eff - float(state.lower_strike), 0.0)

    idx_opt = state.option_index or state.callability_index
    non_neg_index = idx_opt in (
        "WorstOf", "BestOf", "Average", "AverageOf", "Lookback", "TwinWin",
        "RatioAB", "RatioBA", "RatioBestWorst", "RatioBestAverage", "RatioAverageWorst",
    )
    assert l_put >= -1e-6
    if non_neg_index and state.m_rem > 0 and not state.already_autocalled:
        assert l_put <= max(float(state.strike), 0.0) * surv_term + 1e-4
        if state.lower_strike is not None and float(state.lower_strike) <= float(state.strike):
            assert l_put <= k_eff * surv_term + 1e-4
        if float(state.participation_call_rate) == 0.0 and float(state.one_star_coupon) == 0.0:
            df_last = float(state.df_pay[-1].item())
            assert abs(option_pv) <= float(state.option_gearing) * k_eff * df_last + 1e-4
            assert option_pv <= 1e-6


@dataclass
class _ScenarioCase:
    sc_id: str
    category: str
    name: str
    req: Any = None
    expect_schema_reject: bool = False
    expect_runtime_reject: bool = False
    expected_reject_substr: str = ""
    expect_fallback: Optional[bool] = None
    expected_fb_reasons: tuple[str, ...] = ()
    custom_check: Optional[Callable[[Any], None]] = None
    special_fn: Optional[Callable[[Any], None]] = None


def _build_scenario_registry() -> Dict[str, _ScenarioCase]:
    cases: Dict[str, _ScenarioCase] = {}
    t0_base = utc_dt(2026, 6, 15)
    t0_s = utc_dt(2026, 1, 15)
    tks3 = ["EQ00_RUB", "EQ01_RUB", "EQ02_RUB"]

    def reg(case: _ScenarioCase) -> None:
        cases[case.sc_id] = case

    # Group A: SC-001 .. SC-015
    reg(_ScenarioCase("SC-001", "Basket", "Reference fixture eq_autocallable_request.txt",
                      req=lambda: parse_murex_autocall_dag(REQUEST_TXT_PATH)[0], expect_fallback=False))
    reg(_ScenarioCase("SC-002", "Basket", "D=1 single-asset", req=make_synthetic_request_dict(d_assets=1), expect_fallback=False))
    reg(_ScenarioCase("SC-003", "Basket", "D=2 two-asset", req=make_synthetic_request_dict(d_assets=2)))
    reg(_ScenarioCase("SC-004", "Basket", "D=5 five-asset", req=make_synthetic_request_dict(d_assets=5)))
    reg(_ScenarioCase("SC-005", "Basket", "D=8 max in-domain", req=make_synthetic_request_dict(d_assets=8)))
    reg(_ScenarioCase("SC-006", "Basket", "D=9 exceeding d_max=8", req=make_synthetic_request_dict(d_assets=9),
                      expect_fallback=True, expected_fb_reasons=("BASKET_CARDINALITY_EXCEEDED",)))
    reg(_ScenarioCase("SC-007", "Basket", "D=12 exceeding d_max=8", req=make_synthetic_request_dict(d_assets=12),
                      expect_fallback=True, expected_fb_reasons=("BASKET_CARDINALITY_EXCEEDED",)))

    def _sc008_check(ctx: dict[str, Any]) -> None:
        rt_res = ctx.get("rt_res")
        if rt_res is not None and isinstance(rt_res.Delta, dict):
            vals = list(rt_res.Delta.values())
            assert max(vals) - min(vals) < 1e-4

    reg(_ScenarioCase("SC-008", "Basket", "Tied spots S_i=100 equal vol/corr D=5",
                      req=make_synthetic_request_dict(d_assets=5, spots=[100.0] * 5, ref_spots=[100.0] * 5, vol_level=0.30, corr_val=0.50),
                      custom_check=_sc008_check))
    reg(_ScenarioCase("SC-009", "Basket", "Identical ref spots, disparate spots",
                      req=make_synthetic_request_dict(d_assets=5, spots=[20.0, 60.0, 100.0, 140.0, 200.0], ref_spots=[100.0] * 5)))
    reg(_ScenarioCase("SC-010", "Basket", "Tied spots near-singular corr 0.9999",
                      req=make_synthetic_request_dict(d_assets=5, spots=[100.0] * 5, corr_val=0.9999)))
    reg(_ScenarioCase("SC-011", "Basket", "Zero spot validation",
                      req=make_synthetic_request_dict(d_assets=3, spots=[0.0, 100.0, 100.0]),
                      expect_schema_reject=True, expected_reject_substr="strictly positive"))
    reg(_ScenarioCase("SC-012", "Basket", "Negative spot validation",
                      req=make_synthetic_request_dict(d_assets=3, spots=[-10.0, 100.0, 100.0]),
                      expect_schema_reject=True, expected_reject_substr="strictly positive"))
    reg(_ScenarioCase("SC-013", "Basket", "Zero reference spot validation",
                      req=make_synthetic_request_dict(d_assets=3, ref_spots=[0.0, 100.0, 100.0]),
                      expect_schema_reject=True, expected_reject_substr="strictly positive"))
    reg(_ScenarioCase("SC-014", "Basket", "Negative reference spot validation",
                      req=make_synthetic_request_dict(d_assets=3, ref_spots=[-50.0, 100.0, 100.0]),
                      expect_schema_reject=True, expected_reject_substr="strictly positive"))

    def _sc015_perm(rt: Any) -> None:
        r_a = EQAutocallableTradeRequest.model_validate(
            make_synthetic_request_dict(d_assets=3, spots=[85.0, 105.0, 120.0], ref_spots=[90.0, 100.0, 110.0], vol_level=0.35)
        )
        r_b = copy.deepcopy(r_a)
        r_b.Product.UnderlyingIdentifiers = list(reversed(r_a.Product.UnderlyingIdentifiers))
        r_b.Product.UnderlyingReferenceSpots = list(reversed(r_a.Product.UnderlyingReferenceSpots))
        st_a = preprocess_eq_autocall(r_a)
        st_b = preprocess_eq_autocall(r_b)
        assert float(st_a.baseline.product_value_proxy.item()) == pytest.approx(
            float(st_b.baseline.product_value_proxy.item()), abs=1e-12
        )
        if rt is not None:
            pa = rt.price_trade(r_a, engine="auto", include_all_greeks=False)
            pb = rt.price_trade(r_b, engine="auto", include_all_greeks=False)
            assert abs(pa.ProductValue - pb.ProductValue) < 1e-6
            ga = rt.price_trade(r_a, engine="surrogate", include_all_greeks=True)
            gb = rt.price_trade(r_b, engine="surrogate", include_all_greeks=True)
            assert abs(ga.ProductValue - gb.ProductValue) < 1e-6
            assert ga.Delta is not None and gb.Delta is not None
            for k in ga.Delta:
                assert abs(ga.Delta[k] - gb.Delta[k]) < 1e-5

    reg(_ScenarioCase("SC-015", "Basket", "Asset permutation invariance", special_fn=_sc015_perm))

    # Group B: SC-016 .. SC-033
    obs_past = [utc_dt(2025, 6, 15), utc_dt(2025, 9, 15), utc_dt(2025, 12, 15), utc_dt(2026, 3, 15)]
    pay_past = [d + datetime.timedelta(days=3) for d in obs_past]
    reg(_ScenarioCase("SC-016", "Schedule", "M_rem=0 all settled",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=utc_dt(2025, 3, 15),
                                                      obs_dates=obs_past, pay_dates=pay_past,
                                                      fixings_override=_make_past_fixings(tks3, obs_past, [[80.0] * 4] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_zero(ctx["pv_proxy"])))

    obs_it = [utc_dt(2025, 9, 15), utc_dt(2025, 12, 15), utc_dt(2026, 3, 15), t0_base - datetime.timedelta(days=1)]
    pay_it = [d + datetime.timedelta(days=3) for d in obs_it[:-1]] + [t0_base + datetime.timedelta(days=1)]
    reg(_ScenarioCase("SC-017", "Schedule", "M_rem=0 in-transit final above strike",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=utc_dt(2025, 6, 15),
                                                      obs_dates=obs_it, pay_dates=pay_it, fixed_cpn_amt=0.05, cpn_levels=[0.75] * 4,
                                                      fixings_override=_make_past_fixings(tks3, obs_it, [[95.0, 95.0, 95.0, 95.0]] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_gt(ctx["pv_proxy"], 0.90)))
    reg(_ScenarioCase("SC-018", "Schedule", "M_rem=0 in-transit final ITM put W=0.50",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=utc_dt(2025, 6, 15),
                                                      obs_dates=obs_it, pay_dates=pay_it, fixed_cpn_amt=0.05, cpn_levels=[0.75] * 4,
                                                      strike=0.90, option_gearing=1.0,
                                                      fixings_override=_make_past_fixings(tks3, obs_it, [[80.0, 80.0, 80.0, 50.0]] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_between(ctx["pv_proxy"], 0.50, 0.70)))

    obs_eq = [utc_dt(2025, 9, 15), utc_dt(2025, 12, 15), utc_dt(2026, 3, 15), t0_base]
    pay_eq = [d + datetime.timedelta(days=3) for d in obs_eq[:-1]] + [t0_base + datetime.timedelta(days=2)]
    reg(_ScenarioCase("SC-019", "Schedule", "AsOfDate == ObsDate (tau=0d, PayDate=t0+2d)",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=utc_dt(2025, 6, 15),
                                                      obs_dates=obs_eq, pay_dates=pay_eq,
                                                      fixings_override=_make_past_fixings(tks3, obs_eq, [[80.0, 80.0, 80.0, 95.0]] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_gt(ctx["pv_proxy"], 0.90)))

    pay_today = [d + datetime.timedelta(days=3) for d in obs_eq[:-1]] + [t0_base]
    reg(_ScenarioCase("SC-020", "Schedule", "AsOfDate == ObsDate == PayDate settles today",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=utc_dt(2025, 6, 15),
                                                      obs_dates=obs_eq, pay_dates=pay_today,
                                                      fixings_override=_make_past_fixings(tks3, obs_eq, [[80.0, 80.0, 80.0, 95.0]] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_gt(ctx["state"].seasoned_state.cash_to_settle_today, 0.90)))

    reg(_ScenarioCase("SC-021", "Schedule", "M_rem=1 tau=1h",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, obs_dates=[t0_base + datetime.timedelta(hours=1)],
                                                      pay_dates=[t0_base + datetime.timedelta(days=3)])))
    reg(_ScenarioCase("SC-022", "Schedule", "M_rem=1 tau=1d",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, obs_dates=[t0_base + datetime.timedelta(days=1)],
                                                      pay_dates=[t0_base + datetime.timedelta(days=3)])))
    obs_pd = [t0_base - datetime.timedelta(days=3), t0_base + datetime.timedelta(days=90)]
    pay_pd = [t0_base, t0_base + datetime.timedelta(days=93)]
    reg(_ScenarioCase("SC-023", "Schedule", "AsOfDate == PayDate on past period + 1 future",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=t0_base - datetime.timedelta(days=90),
                                                      obs_dates=obs_pd, pay_dates=pay_pd,
                                                      fixings_override=_make_past_fixings(tks3, [obs_pd[0]], [[80.0]] * 3))))

    reg(_ScenarioCase("SC-024", "Schedule", "M_rem=48 exact m_max",
                      req=make_synthetic_request_dict(d_assets=3, m_periods=48,
                                                      obs_dates=[t0_s + datetime.timedelta(days=30 * (k + 1)) for k in range(48)],
                                                      pay_dates=[t0_s + datetime.timedelta(days=30 * (k + 1) + 3) for k in range(48)])))
    reg(_ScenarioCase("SC-025", "Schedule", "M_rem=60 exceeding m_max=48",
                      req=make_synthetic_request_dict(d_assets=3, m_periods=60,
                                                      obs_dates=[t0_s + datetime.timedelta(days=30 * (k + 1)) for k in range(60)],
                                                      pay_dates=[t0_s + datetime.timedelta(days=30 * (k + 1) + 3) for k in range(60)]),
                      expect_fallback=True, expected_fb_reasons=("SCHEDULE_HORIZON_EXCEEDED",)))

    obs_irr = [t0_base + datetime.timedelta(days=7), t0_base + datetime.timedelta(days=187), t0_base + datetime.timedelta(days=197)]
    reg(_ScenarioCase("SC-026", "Schedule", "Irregular stubs 7d front 180d body 10d back",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, obs_dates=obs_irr,
                                                      pay_dates=[d + datetime.timedelta(days=3) for d in obs_irr])))

    obs_ko12 = [t0_base + datetime.timedelta(days=90 * (k - 3)) for k in range(12)]
    obs_ko12[3] = t0_base - datetime.timedelta(days=10)
    reg(_ScenarioCase("SC-027", "Schedule", "Already knocked-out in Period 3 settled",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=obs_ko12[0] - datetime.timedelta(days=90),
                                                      obs_dates=obs_ko12, pay_dates=[d + datetime.timedelta(days=3) for d in obs_ko12],
                                                      ko_levels=[1.00] * 11 + [500.0],
                                                      fixings_override=_make_past_fixings(tks3, obs_ko12[:4], [[85.0, 90.0, 105.0, 92.0]] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_already_ko(ctx["state"], ctx["pv_proxy"], zero_pv=True)))

    obs_ko_it = [t0_base - datetime.timedelta(days=90), t0_base - datetime.timedelta(days=1), t0_base + datetime.timedelta(days=90)]
    pay_ko_it = [t0_base - datetime.timedelta(days=87), t0_base + datetime.timedelta(days=1), t0_base + datetime.timedelta(days=93)]
    reg(_ScenarioCase("SC-028", "Schedule", "Already knocked-out in IN-TRANSIT period",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=t0_base - datetime.timedelta(days=180),
                                                      obs_dates=obs_ko_it, pay_dates=pay_ko_it, ko_levels=[1.00, 1.00, 500.0],
                                                      fixed_cpn_amt=0.06,
                                                      fixings_override=_make_past_fixings(tks3, obs_ko_it[:2], [[85.0, 110.0]] * 3)),
                      expect_fallback=True, expected_fb_reasons=("ZERO_FUTURE_PERIODS",),
                      custom_check=lambda ctx: _assert_already_ko(ctx["state"], ctx["pv_proxy"], zero_pv=False)))

    obs_mem = [t0_base + datetime.timedelta(days=90 * (k - 10)) for k in range(12)]
    obs_mem[10] = t0_base - datetime.timedelta(days=10)
    reg(_ScenarioCase("SC-029", "Schedule", "Memory coupon snowball 11 unpaid periods",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=obs_mem[0] - datetime.timedelta(days=90),
                                                      obs_dates=obs_mem, pay_dates=[d + datetime.timedelta(days=3) for d in obs_mem],
                                                      ko_levels=[1.00] * 11 + [500.0], cpn_levels=[0.75] * 12,
                                                      is_memory=True, fixed_cpn_amt=0.05,
                                                      fixings_override=_make_past_fixings(tks3, obs_mem[:11], [[60.0] * 11] * 3)),
                      custom_check=lambda ctx: _assert_approx(ctx["state"].c_mem_0, 0.55)))

    obs_ki = [t0_base - datetime.timedelta(days=90), t0_base + datetime.timedelta(days=90), t0_base + datetime.timedelta(days=180)]
    reg(_ScenarioCase("SC-030", "Schedule", "Already knocked-in KI history W=0.50 < B_KI=0.75",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=t0_base - datetime.timedelta(days=180),
                                                      obs_dates=obs_ki, pay_dates=[d + datetime.timedelta(days=3) for d in obs_ki],
                                                      ki_barrier=0.75, spots=[85.0, 85.0, 85.0],
                                                      fixings_override=_make_past_fixings(tks3, [obs_ki[0]], [[50.0]] * 3)),
                      custom_check=lambda ctx: _assert_approx(ctx["state"].i_ki_past, 1.0)))

    obs_pin = [t0_base + datetime.timedelta(days=5), t0_base + datetime.timedelta(days=95)]
    reg(_ScenarioCase("SC-031", "Schedule", "Pin-risk KO tau=5d W_0=1.00",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, obs_dates=obs_pin,
                                                      pay_dates=[d + datetime.timedelta(days=3) for d in obs_pin],
                                                      spots=[100.0, 100.0, 100.0], ko_levels=[1.00, 500.0]),
                      expect_fallback=True, expected_fb_reasons=("IMMINENT_BARRIER_PIN_RISK",)))
    reg(_ScenarioCase("SC-032", "Schedule", "Pin-risk Coupon tau=5d W_0=0.75",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, obs_dates=obs_pin,
                                                      pay_dates=[d + datetime.timedelta(days=3) for d in obs_pin],
                                                      spots=[75.0, 75.0, 75.0], ko_levels=[500.0, 500.0], cpn_levels=[0.75, 0.75]),
                      expect_fallback=True, expected_fb_reasons=("IMMINENT_BARRIER_PIN_RISK",)))
    obs_pin_ki = [t0_base + datetime.timedelta(days=5)]
    reg(_ScenarioCase("SC-033", "Schedule", "Pin-risk KI tau=5d W_0=0.70",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, m_periods=1, obs_dates=obs_pin_ki,
                                                      pay_dates=[obs_pin_ki[0] + datetime.timedelta(days=3)],
                                                      spots=[70.0, 70.0, 70.0], ko_levels=[500.0], ki_barrier=0.70),
                      expect_fallback=True, expected_fb_reasons=("IMMINENT_BARRIER_PIN_RISK",)))

    # Group C: SC-034 .. SC-068
    reg(_ScenarioCase("SC-034", "Market", "Bankrupt 1 of 5", req=make_synthetic_request_dict(d_assets=5, spots=[0.1, 100.0, 100.0, 100.0, 100.0]), expect_fallback=True))
    reg(_ScenarioCase("SC-035", "Market", "All bankrupt", req=make_synthetic_request_dict(d_assets=5, spots=[0.1] * 5), expect_fallback=True))
    reg(_ScenarioCase("SC-036", "Market", "Moonshot 1 of 5", req=make_synthetic_request_dict(d_assets=5, spots=[5000.0, 100.0, 100.0, 100.0, 100.0]), expect_fallback=True))
    reg(_ScenarioCase("SC-037", "Market", "All moonshot", req=make_synthetic_request_dict(d_assets=5, spots=[5000.0] * 5), expect_fallback=True))
    reg(_ScenarioCase("SC-038", "Market", "Mixed bankrupt + moonshot", req=make_synthetic_request_dict(d_assets=5, spots=[0.1, 5000.0, 100.0, 100.0, 100.0]), expect_fallback=True))
    reg(_ScenarioCase("SC-039", "Market", "Ultra-low vol 0.005 clamped", req=make_synthetic_request_dict(d_assets=3, vol_level=0.005)))
    reg(_ScenarioCase("SC-040", "Market", "Boundary low vol 0.01", req=make_synthetic_request_dict(d_assets=3, vol_level=0.01)))
    reg(_ScenarioCase("SC-041", "Market", "Crisis vol 2.50", req=make_synthetic_request_dict(d_assets=3, vol_level=2.50), expect_fallback=True))
    reg(_ScenarioCase("SC-042", "Market", "Extreme crisis vol 4.50 3Y",
                      req=make_synthetic_request_dict(d_assets=3, vol_level=4.50,
                                                      obs_dates=[t0_s + datetime.timedelta(days=270 * (k + 1)) for k in range(4)]),
                      expect_fallback=True))
    reg(_ScenarioCase("SC-043", "Market", "Uncertifiable vol 6.00 -> SMILE_ARBITRAGE",
                      req=make_synthetic_request_dict(d_assets=3, vol_level=6.00),
                      expect_runtime_reject=True, expected_reject_substr="SMILE_ARBITRAGE"))
    reg(_ScenarioCase("SC-044", "Market", "Negative vol -0.10 -> SMILE_ARBITRAGE",
                      req=make_synthetic_request_dict(d_assets=3, vol_level=-0.10),
                      expect_runtime_reject=True, expected_reject_substr="SMILE_ARBITRAGE"))
    reg(_ScenarioCase("SC-045", "Market", "NaN vol -> schema reject",
                      req=make_synthetic_request_dict(d_assets=3, vol_level=float("nan")),
                      expect_schema_reject=True, expected_reject_substr="finite"))

    vg_inv = [[0.95] * 7, [0.60] * 7, [0.35] * 7, [0.20] * 7, [0.12] * 7]
    reg(_ScenarioCase("SC-046", "Market", "Inverted vol term structure -> projected",
                      req=make_synthetic_request_dict(d_assets=3, exp_days=[30, 90, 180, 365, 730], vol_grid_override=vg_inv)))
    vg_zc = [[0.40] * 7, [0.40 / math.sqrt(2.0)] * 7, [0.20] * 7]
    reg(_ScenarioCase("SC-047", "Market", "Zero calendar variance dw/dT=0",
                      req=make_synthetic_request_dict(d_assets=3, exp_days=[180, 360, 720], vol_grid_override=vg_zc)))
    # Arbitrage-free Gatheral SVI slice with extreme negative skew rho=-0.85 (passes Durrleman, triggers OOD)
    svi_rho = -0.85
    svi_phi = 1.4
    vg_skew = [
        [
            math.sqrt(
                max(
                    0.09
                    * 0.5
                    * (
                        1.0
                        + svi_rho * svi_phi * math.log(sm)
                        + math.sqrt((svi_phi * math.log(sm) + svi_rho) ** 2 + (1.0 - svi_rho ** 2))
                    ),
                    0.01,
                )
            )
            for sm in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
        ]
        for _ in range(5)
    ]
    reg(_ScenarioCase("SC-048", "Market", "Extreme arbitrage-free SVI skew smile rho=-0.85",
                      req=make_synthetic_request_dict(d_assets=3, exp_days=[30, 90, 180, 365, 730], vol_grid_override=vg_skew),
                      expect_fallback=True))
    vg_bfly = [[0.20, 0.95, 0.12, 0.95, 0.12, 0.95, 0.20] for _ in range(5)]
    reg(_ScenarioCase("SC-049", "Market", "Extreme sawtooth butterfly -> SMILE_ARBITRAGE",
                      req=make_synthetic_request_dict(d_assets=3, exp_days=[30, 90, 180, 365, 730], vol_grid_override=vg_bfly),
                      expect_runtime_reject=True, expected_reject_substr="SMILE_ARBITRAGE"))

    reg(_ScenarioCase("SC-050", "Market", "Duplicate strikes",
                      req=make_synthetic_request_dict(d_assets=3, strikes_mult=[0.5, 1.0, 1.0, 1.0, 1.5],
                                                      exp_days=[30, 180, 365, 730], vol_grid_override=[[0.30, 0.28, 0.29, 0.31, 0.30]] * 4)))
    reg(_ScenarioCase("SC-051", "Market", "Unsorted reversed strikes",
                      req=make_synthetic_request_dict(d_assets=3, strikes_mult=[1.5, 1.0, 0.5],
                                                      exp_days=[30, 180, 365, 730], vol_grid_override=[[0.30, 0.28, 0.32]] * 4)))
    reg(_ScenarioCase("SC-052", "Market", "Negative rate r=-3%", req=make_synthetic_request_dict(d_assets=3, rate_pct=-3.0)))
    reg(_ScenarioCase("SC-053", "Market", "Hyperinflation rate r=+50%", req=make_synthetic_request_dict(d_assets=3, rate_pct=50.0)))
    reg(_ScenarioCase("SC-054", "Market", "Inverted yield curve 30% -> -2%",
                      req=make_synthetic_request_dict(d_assets=3, rate_curve_pcts=[30.0, 22.0, 14.0, 8.0, 4.0, 1.0, -1.0, -2.0])))
    reg(_ScenarioCase("SC-055", "Market", "Negative repo spread -10%", req=make_synthetic_request_dict(d_assets=3, repo_pct=-10.0)))
    reg(_ScenarioCase("SC-056", "Market", "High positive repo spread +30%", req=make_synthetic_request_dict(d_assets=3, repo_pct=30.0)))
    reg(_ScenarioCase("SC-057", "Market", "Mixed repo curves omitted/-10%/+30%", req=make_synthetic_request_dict(d_assets=3, mixed_repos=[None, -10.0, 30.0])))
    reg(_ScenarioCase("SC-058", "Market", "Huge cash dividend 0.85*S0",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_s, div_events_per_asset={0: [(t0_s + datetime.timedelta(days=45), t0_s + datetime.timedelta(days=50), 85.0)]}),
                      expect_fallback=True, expected_fb_reasons=("MAHALANOBIS_OOD",)))
    reg(_ScenarioCase("SC-059", "Market", "Dividend exceeding spot 1.25*S0",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_s, div_events_per_asset={0: [(t0_s + datetime.timedelta(days=45), t0_s + datetime.timedelta(days=50), 125.0)]}),
                      expect_fallback=True, expected_fb_reasons=("MAHALANOBIS_OOD",)))
    reg(_ScenarioCase("SC-060", "Market", "Dividend ex_date == t0 excluded",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_s, div_events_per_asset={0: [(t0_s, t0_s + datetime.timedelta(days=5), 25.0)]}),
                      custom_check=lambda ctx: _assert_approx(float(ctx["state"].dividend_schedule.div_pv[0, -1].item()), 0.0)))
    obs_d4 = [t0_s + datetime.timedelta(days=90 * (k + 1)) for k in range(4)]
    reg(_ScenarioCase("SC-061", "Market", "Dividend ex_date == T included",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_s, obs_dates=obs_d4, div_events_per_asset={0: [(obs_d4[-1], obs_d4[-1] + datetime.timedelta(days=5), 10.0)]}),
                      custom_check=lambda ctx: _assert_gt(float(ctx["state"].dividend_schedule.div_pv[0, -1].item()), 5.0)))
    reg(_ScenarioCase("SC-062", "Market", "Dividend ex_date == T+1d excluded",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_s, obs_dates=obs_d4, div_events_per_asset={0: [(obs_d4[-1] + datetime.timedelta(days=1), obs_d4[-1] + datetime.timedelta(days=6), 15.0)]}),
                      custom_check=lambda ctx: _assert_approx(float(ctx["state"].dividend_schedule.div_pv[0, -1].item()), 0.0)))
    reg(_ScenarioCase("SC-063", "Market", "PredefinedDividends overrides MarketDataSet",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_s, div_events_per_asset={0: [(t0_s + datetime.timedelta(days=45), t0_s + datetime.timedelta(days=50), 2.0)]},
                                                      predefined_divs={"EQ00_RUB": {"ExDates": [t0_s + datetime.timedelta(days=45)], "Amounts": [25.0]}}),
                      custom_check=lambda ctx: _assert_gt(float(ctx["state"].dividend_schedule.div_pv[0, -1].item()), 20.0)))
    corr_nonpsd = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]], dtype=np.float64)
    reg(_ScenarioCase("SC-064", "Market", "Non-PSD correlation D=3",
                      req=make_synthetic_request_dict(d_assets=3, corr_matrix=corr_nonpsd),
                      expect_fallback=True, expected_fb_reasons=("CORRELATION_NOT_PSD",)))
    reg(_ScenarioCase("SC-065", "Market", "Near-perfect corr 0.9999 D=5", req=make_synthetic_request_dict(d_assets=5, corr_val=0.9999)))
    reg(_ScenarioCase("SC-066", "Market", "Near-perfect neg corr -0.99 D=2", req=make_synthetic_request_dict(d_assets=2, corr_val=-0.99)))
    reg(_ScenarioCase("SC-067", "Market", "Equi-correlated non-PSD D=5 rho=-0.30",
                      req=make_synthetic_request_dict(d_assets=5, corr_val=-0.30),
                      expect_fallback=True, expected_fb_reasons=("CORRELATION_NOT_PSD",)))
    reg(_ScenarioCase("SC-068", "Market", "Rank-1 corr rho=1.0 D=3", req=make_synthetic_request_dict(d_assets=3, corr_val=1.0)))

    # Group D: SC-069 .. SC-089 and SC-132 .. SC-136 (All 18 monitoring indices & aliases)
    for sc_id, idx_name in _MONITORING_ALL:
        d_use = 2 if ("AB" in idx_name or "BA" in idx_name or idx_name in ("Outperformance", "Spread")) else 3
        is_std = idx_name in ("WorstOf", "BestOf", "Average", "AverageOf", "EquallyWeightedBasket", "MultiPerf")
        reg(_ScenarioCase(
            sc_id,
            "Index",
            f"MonitoringIndex={idx_name}",
            req=make_synthetic_request_dict(d_assets=d_use, spots=[85.0, 115.0] if d_use == 2 else [75.0, 100.0, 125.0], callability_index=idx_name),
            expect_fallback=(None if idx_name == "WorstOf" else (False if is_std else True)),
            expected_fb_reasons=(() if is_std else ("EXOTIC_MONITORING_INDEX",)),
        ))

    reg(_ScenarioCase("SC-089", "Index", "Mixed monitoring indices WoF/BoF/Avg",
                      req=make_synthetic_request_dict(d_assets=3, spots=[65.0, 100.0, 130.0], callability_index="WorstOf", coupon_index="BestOf", option_index="Average"),
                      expect_fallback=True, expected_fb_reasons=("EXOTIC_MONITORING_INDEX",)))

    # Group E: SC-090 .. SC-131
    reg(_ScenarioCase("SC-090", "Exotic", "Coupon Out Up", req=make_synthetic_request_dict(d_assets=3, coupon_barrier_type="Out", coupon_barrier_direction="Up", cpn_levels=[0.95] * 4), expect_fallback=True))
    reg(_ScenarioCase("SC-091", "Exotic", "Coupon In Down", req=make_synthetic_request_dict(d_assets=3, coupon_barrier_type="In", coupon_barrier_direction="Down", cpn_levels=[0.95] * 4), expect_fallback=True))
    reg(_ScenarioCase("SC-092", "Exotic", "Coupon Out Down", req=make_synthetic_request_dict(d_assets=3, coupon_barrier_type="Out", coupon_barrier_direction="Down", cpn_levels=[0.85] * 4), expect_fallback=True))
    obs_out_s = [t0_base - datetime.timedelta(days=90), t0_base + datetime.timedelta(days=90)]
    reg(_ScenarioCase("SC-093", "Exotic", "Seasoned Coupon Out Up memory accumulation",
                      req=make_synthetic_request_dict(d_assets=3, as_of=t0_base, start_date=t0_base - datetime.timedelta(days=180),
                                                      obs_dates=obs_out_s, pay_dates=[d + datetime.timedelta(days=3) for d in obs_out_s],
                                                      coupon_barrier_type="Out", coupon_barrier_direction="Up",
                                                      cpn_levels=[0.80, 0.80], is_memory=True, fixed_cpn_amt=0.05,
                                                      fixings_override=_make_past_fixings(tks3, [obs_out_s[0]], [[95.0]] * 3)),
                      expect_fallback=True,
                      custom_check=lambda ctx: _assert_approx(ctx["state"].c_mem_0, 0.05)))
    reg(_ScenarioCase("SC-094", "Sentinel", "Sentinel KO=500.0", req=make_synthetic_request_dict(d_assets=3, ko_levels=[500.0] * 4)))
    reg(_ScenarioCase("SC-095", "Sentinel", "Sentinel KO=65535.0", req=make_synthetic_request_dict(d_assets=3, ko_levels=[65535.0] * 4)))
    reg(_ScenarioCase("SC-096", "Sentinel", "Sentinel Coupon=1e-5", req=make_synthetic_request_dict(d_assets=3, cpn_levels=[1e-5] * 4, spots=[40.0] * 3)))
    reg(_ScenarioCase("SC-097", "Sentinel", "Sentinel Coupon=0.0", req=make_synthetic_request_dict(d_assets=3, cpn_levels=[0.0] * 4, spots=[40.0] * 3)))
    reg(_ScenarioCase("SC-098", "Exotic", "LowerStrike == Strike -> zero option",
                      req=make_synthetic_request_dict(d_assets=3, strike=0.90, lower_strike=0.90, spots=[40.0] * 3),
                      custom_check=lambda ctx: _assert_approx(ctx["l_put_proxy"], 0.0)))
    reg(_ScenarioCase("SC-099", "Exotic", "LowerStrike > Strike -> schema reject",
                      req=make_synthetic_request_dict(d_assets=3, strike=0.90, lower_strike=1.10), expect_schema_reject=True))
    reg(_ScenarioCase("SC-100", "Exotic", "Put spread K=0.90 K_low=0.50 deep ITM",
                      req=make_synthetic_request_dict(d_assets=3, strike=0.90, lower_strike=0.50, spots=[30.0] * 3),
                      custom_check=lambda ctx: _assert_le(ctx["l_put_proxy"], 0.40 + 1e-4)))
    reg(_ScenarioCase("SC-101", "Exotic", "OptionGearing=0.0",
                      req=make_synthetic_request_dict(d_assets=3, option_gearing=0.0, spots=[40.0] * 3),
                      custom_check=lambda ctx: _assert_approx(ctx["option_pv_proxy"], 0.0)))
    reg(_ScenarioCase("SC-102", "Exotic", "OptionGearing=5.0", req=make_synthetic_request_dict(d_assets=3, option_gearing=5.0, spots=[75.0] * 3)))
    reg(_ScenarioCase("SC-103", "Exotic", "NoNotionalRepayment=True",
                      req=make_synthetic_request_dict(d_assets=3, no_notional_repayment=True),
                      custom_check=lambda ctx: _assert_approx(ctx["notional_pv_proxy"], 0.0)))
    reg(_ScenarioCase("SC-104", "Exotic", "AutoCallCoupon=0.15", req=make_synthetic_request_dict(d_assets=3, autocall_coupon=0.15)))
    reg(_ScenarioCase("SC-105", "Exotic", "ForceCouponsOnAutocall=True", req=make_synthetic_request_dict(d_assets=3, ko_levels=[0.90, 0.90, 0.90, 500.0], cpn_levels=[1.15] * 4, force_coupons_on_autocall=True)))
    reg(_ScenarioCase("SC-106", "Exotic", "ForceCouponsOnAutocall=False", req=make_synthetic_request_dict(d_assets=3, ko_levels=[0.90, 0.90, 0.90, 500.0], cpn_levels=[1.15] * 4, force_coupons_on_autocall=False)))
    reg(_ScenarioCase("SC-107", "Exotic", "AllowAutocallOnExpiry=False non-sentinel B_KO", req=make_synthetic_request_dict(d_assets=3, ko_levels=[0.90] * 4, allow_autocall_on_expiry=False)))
    reg(_ScenarioCase("SC-108", "Exotic", "AllowAutocallOnExpiry=True final B_KO=0.90",
                      req=make_synthetic_request_dict(d_assets=3, ko_levels=[0.90] * 4, allow_autocall_on_expiry=True, autocall_coupon=0.10, spots=[110.0] * 3),
                      expect_fallback=True))
    reg(_ScenarioCase("SC-109", "Exotic", "IsLongBasket=False", req=make_synthetic_request_dict(d_assets=3, is_long_basket=False, spots=[60.0] * 3)))
    reg(_ScenarioCase("SC-110", "Exotic", "IsLongBasket=True", req=make_synthetic_request_dict(d_assets=3, is_long_basket=True, spots=[60.0] * 3)))
    reg(_ScenarioCase("SC-111", "Exotic", "OneStar only", req=make_synthetic_request_dict(d_assets=3, one_star_strike=1.10, one_star_coupon=0.05), expect_fallback=True))
    reg(_ScenarioCase("SC-112", "Exotic", "ParticipationCall only", req=make_synthetic_request_dict(d_assets=3, part_call_strike=1.00, part_call_rate=1.50), expect_fallback=True))
    reg(_ScenarioCase("SC-113", "Exotic", "OneStar + ParticipationCall", req=make_synthetic_request_dict(d_assets=3, one_star_strike=1.10, one_star_coupon=0.05, part_call_strike=1.00, part_call_rate=1.00), expect_fallback=True))

    fl_dates = [t0_s + datetime.timedelta(days=90 * (k + 1) + 3) for k in range(4)]
    reg(_ScenarioCase("SC-114", "Funding", "FloatingFundingLeg +2%", req=make_synthetic_request_dict(d_assets=3, rate_identifier=None, funding_spread=0.02, funding_leg_dates=fl_dates)))
    reg(_ScenarioCase("SC-115", "Funding", "FloatingFundingLeg -5%", req=make_synthetic_request_dict(d_assets=3, rate_identifier=None, funding_spread=-0.05, funding_leg_dates=fl_dates)))
    reg(_ScenarioCase("SC-116", "Funding", "FloatingFundingLeg -30%", req=make_synthetic_request_dict(d_assets=3, rate_identifier=None, funding_spread=-0.30, funding_leg_dates=fl_dates)))
    reg(_ScenarioCase("SC-117", "Quanto", "Extreme Quanto rho=+0.95 fx=0.80", req=make_synthetic_request_dict(d_assets=3, asset_ccy_suffix="USD", payment_ccy="RUB", fx_vol=0.80, fx_corr=0.95)))
    reg(_ScenarioCase("SC-118", "Quanto", "Extreme Quanto rho=-0.95 fx=0.80", req=make_synthetic_request_dict(d_assets=3, asset_ccy_suffix="USD", payment_ccy="RUB", fx_vol=0.80, fx_corr=-0.95)))
    reg(_ScenarioCase("SC-119", "Spread", "DigitalSpread=0.0 finite Greeks", req=make_synthetic_request_dict(d_assets=3, ko_spread=0.0, cpn_spread=0.0)))
    reg(_ScenarioCase("SC-120", "Spread", "DigitalSpread=-0.15", req=make_synthetic_request_dict(d_assets=3, ko_spread=-0.15, cpn_spread=-0.15)))
    reg(_ScenarioCase("SC-121", "Spread", "DigitalSpread=+0.15", req=make_synthetic_request_dict(d_assets=3, ko_spread=0.15, cpn_spread=0.15)))
    reg(_ScenarioCase("SC-122", "Exotic", "List-valued FixedCouponAmount", req=make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=[0.01, 0.02, 0.05, 0.10])))

    def _sc123(_: Any) -> None:
        puts = []
        for s_v in (60.0, 80.0, 100.0, 120.0):
            r = EQAutocallableTradeRequest.model_validate(make_synthetic_request_dict(d_assets=3, spots=[s_v] * 3, ko_levels=[500.0] * 4))
            puts.append(float(assemble_autocall_cashflows_and_pv(preprocess_eq_autocall(r)).l_put.item()))
        assert puts[0] > puts[1] > puts[2] > puts[3]

    def _sc124(_: Any) -> None:
        puts = []
        for k_v in (0.70, 0.90, 1.10):
            r = EQAutocallableTradeRequest.model_validate(make_synthetic_request_dict(d_assets=3, strike=k_v, ko_levels=[500.0] * 4))
            puts.append(float(assemble_autocall_cashflows_and_pv(preprocess_eq_autocall(r)).l_put.item()))
        assert puts[0] < puts[1] < puts[2]

    def _sc125(_: Any) -> None:
        p_sums = []
        for ko_v in (0.80, 1.00, 1.20):
            r = EQAutocallableTradeRequest.model_validate(make_synthetic_request_dict(d_assets=3, ko_levels=[ko_v] * 3 + [500.0]))
            p_sums.append(float(assemble_autocall_cashflows_and_pv(preprocess_eq_autocall(r)).p_stop.sum().item()))
        assert p_sums[0] > p_sums[1] > p_sums[2]

    def _sc126(_: Any) -> None:
        r_lo = EQAutocallableTradeRequest.model_validate(make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=0.02))
        r_hi = EQAutocallableTradeRequest.model_validate(make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=0.08))
        assert float(assemble_autocall_cashflows_and_pv(preprocess_eq_autocall(r_hi)).product_value.item()) > float(
            assemble_autocall_cashflows_and_pv(preprocess_eq_autocall(r_lo)).product_value.item()
        )

    def _sc127(_: Any) -> None:
        base_d = make_synthetic_request_dict(d_assets=3)
        pv0 = float(preprocess_eq_autocall(EQAutocallableTradeRequest.model_validate(base_d)).baseline.product_value_proxy.item())
        d_vol = copy.deepcopy(base_d)
        vq = d_vol["Model"]["MarketDataSet"]["VolatilitySurfaces"]["EQ00_RUB"]["VolatilityQuotes"]
        vq[1] = [float(x) + 0.08 for x in vq[1]]
        assert abs(float(preprocess_eq_autocall(EQAutocallableTradeRequest.model_validate(d_vol)).baseline.product_value_proxy.item()) - pv0) > 1e-6

    reg(_ScenarioCase("SC-123", "Invariant", "Spot monotonicity put loss", special_fn=_sc123))
    reg(_ScenarioCase("SC-124", "Invariant", "Strike monotonicity put loss", special_fn=_sc124))
    reg(_ScenarioCase("SC-125", "Invariant", "KO barrier monotonicity sum(p_stop)", special_fn=_sc125))
    reg(_ScenarioCase("SC-126", "Invariant", "Coupon monotonicity", special_fn=_sc126))
    reg(_ScenarioCase("SC-127", "Cache", "Cache collision immunity", special_fn=_sc127))
    reg(_ScenarioCase("SC-128", "Schedule", "FixedRate=0.16 Act365", req=make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=None, fixed_rate=0.16, dcc="Act365")))
    reg(_ScenarioCase("SC-129", "Schedule", "FixedRate=0.16 Business252", req=make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=None, fixed_rate=0.16, dcc="Business252")))
    reg(_ScenarioCase("SC-130", "Schedule", "FixedRate=0.16 Thirty360", req=make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=None, fixed_rate=0.16, dcc="Thirty360")))
    reg(_ScenarioCase("SC-131", "Exotic", "List-valued FixedRate", req=make_synthetic_request_dict(d_assets=3, fixed_cpn_amt=None, fixed_rate=[0.08, 0.10, 0.12, 0.16])))

    assert len(cases) == 136
    return cases


def _assert_zero(x: float) -> None:
    assert abs(x) < 1e-12


def _assert_gt(x: float, lo: float) -> None:
    assert x > lo


def _assert_le(x: float, hi: float) -> None:
    assert x <= hi


def _assert_between(x: float, lo: float, hi: float) -> None:
    assert lo < x < hi


def _assert_approx(x: float, target: float) -> None:
    assert x == pytest.approx(target, abs=1e-10)


def _assert_already_ko(state: AutocallPreprocState, pv: float, *, zero_pv: bool) -> None:
    assert state.seasoned_state.already_autocalled is True
    if zero_pv:
        assert abs(pv) < 1e-12
    else:
        assert pv > 0.90


_SCENARIO_CASES = _build_scenario_registry()
_ALL_SC_IDS = [f"SC-{i:03d}" for i in range(1, 137)]


@pytest.fixture(scope="module")
def autocall_runtime() -> Any:
    try:
        from deepvol.api.eq_autocall_runtime import EQAutocallRuntime

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return EQAutocallRuntime(device=dev, n_fallback_paths=256)
    except Exception:
        return None


@pytest.mark.parametrize("sc_id", _ALL_SC_IDS)
def test_extreme_scenario_matrix(sc_id: str, autocall_runtime: Any) -> None:
    case = _SCENARIO_CASES[sc_id]
    ppp_mod._PREPROC_STATE_CACHE.clear()
    if case.special_fn is not None:
        case.special_fn(autocall_runtime)
        return

    req_src = case.req() if callable(case.req) else case.req
    if isinstance(req_src, EQAutocallableTradeRequest):
        req_model = req_src
    else:
        if case.expect_schema_reject:
            with pytest.raises((ValidationError, ValueError)) as exc_info:
                EQAutocallableTradeRequest.model_validate(req_src)
            if case.expected_reject_substr:
                assert case.expected_reject_substr in str(exc_info.value)
            # Raw dict bypass directly into preprocess_eq_autocall must also reject negative/zero spots
            if sc_id in ("SC-011", "SC-012"):
                with pytest.raises(ValueError):
                    preprocess_eq_autocall(req_src)
            return
        req_model = EQAutocallableTradeRequest.model_validate(req_src)

    if case.expect_runtime_reject:
        with pytest.raises(ValueError) as exc_info:
            preprocess_eq_autocall(req_model)
        if case.expected_reject_substr:
            assert case.expected_reject_substr in str(exc_info.value)
        return

    state = preprocess_eq_autocall(req_model)
    b_out = compute_analytical_autocall_baseline(state)
    proxy_out = assemble_autocall_cashflows_and_pv(state)
    _assert_all_finite(float(b_out.product_value_proxy.item()), f"{sc_id}.proxy")
    _assert_prob_and_put_bounds(
        p_stop=proxy_out.p_stop.detach().cpu().numpy(),
        q_cpn=proxy_out.q_cpn.detach().cpu().numpy(),
        l_put=float(proxy_out.l_put.item()),
        option_pv=float(proxy_out.option_pv.item()),
        state=state,
    )

    qmc_out = price_eq_autocall_dupire_qmc(
        state,
        n_paths=512,
        n_replicates=4,
        mode="dupire_lv",
        n_substeps_per_period=2,
        seed=42,
    )
    _assert_prob_and_put_bounds(
        p_stop=np.asarray(qmc_out.p_stop, dtype=np.float64),
        q_cpn=np.asarray(qmc_out.q_cpn, dtype=np.float64),
        l_put=float(qmc_out.l_put),
        option_pv=float(qmc_out.option_pv),
        state=state,
    )

    rt_res = None
    if autocall_runtime is not None and hasattr(autocall_runtime, "price_trade"):
        want_auto_greeks = sc_id in (
            "SC-001",
            "SC-006",
            "SC-007",
            "SC-016",
            "SC-019",
            "SC-025",
            "SC-027",
            "SC-080",
            "SC-133",
            "SC-134",
            "SC-135",
        )
        rt_res = autocall_runtime.price_trade(req_model, engine="auto", include_all_greeks=want_auto_greeks)
        grk_res = rt_res
        if not want_auto_greeks and state.d_assets <= 8 and state.m_rem <= 48:
            grk_res = autocall_runtime.price_trade(req_model, engine="surrogate", include_all_greeks=True)
        for check_res in (rt_res, grk_res):
            if check_res is rt_res and want_auto_greeks and state.m_rem > 0:
                for g_field in (
                    "Delta",
                    "Gamma",
                    "MultivariateAggregatedCashGamma",
                    "Vega",
                    "Rho",
                    "DividendsSensitivities",
                    "Theta",
                    "Cega",
                ):
                    assert getattr(check_res, g_field) is not None, f"{sc_id}: missing {g_field}"
            _assert_all_finite(
                {
                    "pv": check_res.ProductValue,
                    "se": check_res.StandardError,
                    "opt": check_res.Option,
                    "not": check_res.Notional,
                    "cpn": check_res.Coupon,
                    "fund": check_res.FundingLeg,
                    "delta": check_res.Delta,
                    "gamma": check_res.Gamma,
                    "cash_gamma": check_res.MultivariateAggregatedCashGamma,
                    "vega": check_res.Vega,
                    "rho": check_res.Rho,
                    "divs": check_res.DividendsSensitivities,
                    "theta": check_res.Theta,
                    "cega": check_res.Cega,
                },
                f"{sc_id}.runtime",
            )
        if case.expect_fallback is not None:
            assert bool(rt_res.fallback_triggered) is case.expect_fallback, (
                f"{sc_id}: fallback={rt_res.fallback_triggered}, reasons={rt_res.fallback_reasons}"
            )
        for reason in case.expected_fb_reasons:
            assert reason in rt_res.fallback_reasons, f"{sc_id}: missing {reason} in {rt_res.fallback_reasons}"
        if sc_id == "SC-008":
            rt_res = grk_res

    if case.custom_check is not None:
        case.custom_check(
            {
                "state": state,
                "pv_proxy": float(proxy_out.product_value.item()),
                "l_put_proxy": float(proxy_out.l_put.item()),
                "option_pv_proxy": float(proxy_out.option_pv.item()),
                "notional_pv_proxy": float(proxy_out.notional_pv.item()),
                "qmc": qmc_out,
                "rt_res": rt_res,
            }
        )
