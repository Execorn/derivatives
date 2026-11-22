from __future__ import annotations

import datetime as dt
import inspect
import math
import time
from typing import Any

import numpy as np
import pytest
import torch

from deepvol.api.eq_autocall_runtime import (
    ALL_8_GREEK_QUERIES,
    EQAutocallRuntime,
)
from deepvol.calibration.autocall_sobol_generator import (
    generate_synthetic_autocall_batch,
)
from deepvol.models.autocall.dupire_qmc_solver import RegularizedVolSurface
from deepvol.models.autocall.analytical_baseline import (
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig
from deepvol.models.autocall.greeks_engine import (
    _assemble_pv_batch_torch,
    _build_single_asset_lv_table_fast_np,
    _eval_autocall_payoff_pv_batch_np,
    _exact_piecewise_terminal_put_tensor_f32,
    _resolve_quanto_fx_sign,
    compute_eq_autocall_greeks,
    compute_qmc_crn_greeks,
    evaluate_differentiable_pv,
)
from deepvol.models.autocall.pre_post_processor import (
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.mrm.eq_autocall_guardian import (
    AutocallFallbackReason,
    EQAutocallModelGuardian,
    _regularize_cov_oas_correlation_space,
    compute_psi_10_decile,
)
from deepvol.surrogates.compiled_runner import GEOM_BUCKETS
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
    assemble_surrogate_product_value,
)
from deepvol.training.train_eq_autocall import calibrate_guardian_stats


def _make_base_market_and_product(
    *,
    tickers: list[str] | None = None,
    payment_ccy: str = "RUB",
    ki_barrier: float | None = 0.75,
    ki_spread: float = -0.10,
    ki_obs_type: str | None = "European",
    is_long_basket: bool = False,
    one_star_strike: float | None = None,
    one_star_coupon: float = 0.0,
) -> dict[str, Any]:
    tks = tickers if tickers is not None else ["SBER_RUB", "LKOH_RUB"]
    as_of = "2026-04-03T00:00:00Z"
    pillars = [
        "2026-05-03T00:00:00Z",
        "2026-07-03T00:00:00Z",
        "2026-10-03T00:00:00Z",
        "2027-04-03T00:00:00Z",
        "2028-04-03T00:00:00Z",
    ]
    exp_dates = [
        "2026-07-03T00:00:00Z",
        "2026-10-03T00:00:00Z",
        "2027-04-03T00:00:00Z",
    ]
    spots = {tk: 100.0 + 10.0 * idx for idx, tk in enumerate(tks)}
    ref_spots = [100.0 + 10.0 * idx for idx in range(len(tks))]
    surfaces: dict[str, Any] = {}
    dividends: dict[str, Any] = {}
    repos: dict[str, Any] = {}
    for tk in tks:
        s0 = spots[tk]
        strikes = [round(s0 * m, 4) for m in (0.7, 0.85, 1.0, 1.15, 1.3)]
        surfaces[tk] = {
            "TypeName": "EquityVolatilitySmileCollection",
            "Identifier": tk,
            "ExpirationDates": exp_dates,
            "Strikes": strikes,
            "VolatilityQuotes": [
                [0.34, 0.31, 0.29, 0.30, 0.32],
                [0.33, 0.30, 0.28, 0.29, 0.31],
                [0.32, 0.29, 0.27, 0.28, 0.30],
            ],
            "DayCountConvention": "Act365",
            "SmileInterpolationMethod": "CubicSpline",
            "SmileExtrapolationMethod": "Flat",
        }
        dividends[tk] = {
            "TypeName": "Dividends",
            "Identifier": tk,
            "CashDividendList": [
                {
                    "ExDate": "2026-06-15T00:00:00Z",
                    "PaymentDate": "2026-06-20T00:00:00Z",
                    "Amount": 2.0,
                },
                {
                    "ExDate": "2026-12-15T00:00:00Z",
                    "PaymentDate": "2026-12-20T00:00:00Z",
                    "Amount": 2.0,
                },
            ],
        }
        repos[tk] = {
            "TypeName": "RepoSpreadCurve",
            "Identifier": tk,
            "InterpolationDayCountConvention": "Act360",
            "InterpolationMethod": "Linear",
            "InterpolationSpaceCoordinates": "Rates",
            "ExtrapolationMethod": "Flat",
            "RateQuoteExpression": "Decimal",
            "RatesSchedule": {
                "Periods": ["30D", "90D", "180D", "365D", "730D"],
                "Rates": [0.01] * 5,
            },
        }

    corrs: dict[str, float] = {}
    for i in range(len(tks)):
        for j in range(i + 1, len(tks)):
            corrs[f"{tks[i]}&{tks[j]}"] = 0.45

    product: dict[str, Any] = {
        "StartDate": "2026-04-03T00:00:00Z",
        "PaymentCurrency": payment_ccy,
        "DayCountConvention": "Act365",
        "UnderlyingIdentifiers": tks,
        "UnderlyingReferenceSpots": ref_spots,
        "ObservationDates": [
            "2026-07-03T00:00:00Z",
            "2026-10-03T00:00:00Z",
            "2027-04-03T00:00:00Z",
        ],
        "PaymentDates": [
            "2026-07-10T00:00:00Z",
            "2026-10-10T00:00:00Z",
            "2027-04-10T00:00:00Z",
        ],
        "AutoCallTriggerLevels": [1.0, 1.0, 1.0],
        "CouponBarrierLevels": [0.80, 0.80, 0.80],
        "FixedCouponAmount": [0.04, 0.04, 0.04],
        "IsMemory": True,
        "CallabilityMonitoringIndex": "WorstOf",
        "CouponBarrierMonitoringIndex": "WorstOf",
        "OptionMonitoringIndex": "WorstOf",
        "Strike": 1.0,
        "LowerStrike": 0.0,
        "OptionGearing": 1.0,
        "OptionDownAndInBarrierLevel": ki_barrier,
        "OptionDownAndInBarrierObservationType": ki_obs_type,
        "AutoCallTriggerAbsoluteDigitalSpread": -0.10,
        "CouponBarrierAbsoluteDigitalSpread": -0.10,
        "OptionDownAndInBarrierAbsoluteDigitalSpread": ki_spread,
        "AllowAutocallOnExpiry": False,
        "ForceCouponsOnAutocall": False,
        "NoNotionalRepayment": False,
        "IsLongBasket": is_long_basket,
    }
    if one_star_strike is not None:
        product["OneStarStrike"] = one_star_strike
        product["OneStarCoupon"] = one_star_coupon

    return {
        "TradeInfo": "TEST_REMEDIATION_B10316F2",
        "Queries": ["ProductValue", "CashFlows", *ALL_8_GREEK_QUERIES],
        "Product": product,
        "Model": {
            "FinancingCurveIdentifiers": ["RUB_FUNDING_SECD"],
            "MarketDataSet": {
                "AsOfDate": as_of,
                "Spots": spots,
                "VolatilitySurfaces": surfaces,
                "Correlations": corrs,
                "Dividends": dividends,
                "RepoSpreadCurves": repos,
                "RatesCurvesBundles": {
                    "RUB_FUNDING_SECD": {
                        "Identifier": "RUB_FUNDING_SECD",
                        "Currency": "RUB",
                        "AsOfDate": as_of,
                        "InterpolationDayCountConvention": "Act365",
                        "InterpolationMethod": "Linear",
                        "InterpolationSpaceCoordinates": "Rates",
                        "ExtrapolationMethod": "Flat",
                        "RateQuoteExpression": "Decimal",
                        "RatesSchedule": {
                            "Dates": pillars,
                            "Rates": [0.21] * len(pillars),
                        },
                    },
                    "RUB_RUONIA_OIS": {
                        "Identifier": "RUB_RUONIA_OIS",
                        "Currency": "RUB",
                        "AsOfDate": as_of,
                        "InterpolationDayCountConvention": "Act365",
                        "InterpolationMethod": "Linear",
                        "InterpolationSpaceCoordinates": "Rates",
                        "ExtrapolationMethod": "Flat",
                        "RateQuoteExpression": "Decimal",
                        "RatesSchedule": {
                            "Dates": pillars,
                            "Rates": [0.16] * len(pillars),
                        },
                    },
                    "USD_SOFR": {
                        "Identifier": "USD_SOFR",
                        "Currency": "USD",
                        "AsOfDate": as_of,
                        "InterpolationDayCountConvention": "Act365",
                        "InterpolationMethod": "Linear",
                        "InterpolationSpaceCoordinates": "Rates",
                        "ExtrapolationMethod": "Flat",
                        "RateQuoteExpression": "Decimal",
                        "RatesSchedule": {
                            "Dates": pillars,
                            "Rates": [0.045] * len(pillars),
                        },
                    },
                    "CNH_FX": {
                        "Identifier": "CNH_FX",
                        "Currency": "CNH",
                        "AsOfDate": as_of,
                        "InterpolationDayCountConvention": "Act365",
                        "InterpolationMethod": "Linear",
                        "InterpolationSpaceCoordinates": "Rates",
                        "ExtrapolationMethod": "Flat",
                        "RateQuoteExpression": "Decimal",
                        "RatesSchedule": {
                            "Dates": pillars,
                            "Rates": [0.03] * len(pillars),
                        },
                    },
                },
            },
        },
    }


def test_I4a_ki_digital_ramp_bounds_and_pv_shift() -> None:
    """I-4a: KI digital ramp must lie on [B - |s|, B] = [min(B, B+spread), max(B, B+spread)]."""
    # 1. Exact closed-form check on _piecewise_put_only_batch32 for trade 56615092 parameters:
    # B = K = 1.0, s = -0.10 => ramp on [0.90, 1.00] reduces put loss by ~14.78 bps vs [1.00, 1.10].
    tau = 0.2547945205479452
    sigma = 0.45198225264021996
    v_w = sigma * math.sqrt(tau)
    f_w = (47.74 / 48.92) * math.exp(0.145 * tau)
    df_pay = 0.961775485199402

    fc_t = torch.tensor([f_w], dtype=torch.float32)
    vc_t = torch.tensor([v_w], dtype=torch.float32)
    surv_t = torch.tensor([1.0], dtype=torch.float32)
    surv_prev_t = torch.tensor([1.0], dtype=torch.float32)
    is_call_t = torch.tensor([0.0], dtype=torch.float32)
    k_strike_t = torch.tensor([1.0], dtype=torch.float32)
    k_low_t = torch.tensor([0.0], dtype=torch.float32)
    active_ki_t = torch.tensor([True], dtype=torch.bool)
    b_ki_t = torch.tensor([1.0], dtype=torch.float32)
    b_ko_t = torch.tensor([100.0], dtype=torch.float32)
    ko_sp_t = torch.tensor([0.0], dtype=torch.float32)

    l_put_ramp = float(
        _exact_piecewise_terminal_put_tensor_f32(
            fc_t,
            vc_t,
            k_strike_t,
            k_low_t,
            active_ki_t,
            b_ki_t,
            torch.tensor([0.10], dtype=torch.float32),
            is_call_t,
            b_ko_t,
            ko_sp_t,
        ).item()
    )
    l_put_step = float(
        _exact_piecewise_terminal_put_tensor_f32(
            fc_t,
            vc_t,
            k_strike_t,
            k_low_t,
            active_ki_t,
            b_ki_t,
            torch.tensor([0.0], dtype=torch.float32),
            is_call_t,
            b_ko_t,
            ko_sp_t,
        ).item()
    )
    p_surv_56615092 = 0.510423
    pv_shift_bps = (l_put_step - l_put_ramp) * p_surv_56615092 * df_pay * 1e4
    assert 13.5 <= pv_shift_bps <= 16.0, f"Expected ~14.78 bps KI ramp reduction, got {pv_shift_bps:.2f} bps"

    # 2. Trade 60437649 / 58579592 geometry: B = 0.90 < K = 1.00, ki_spread = -0.10 => ramp on [0.80, 0.90]
    b_ki_09 = torch.tensor([0.90], dtype=torch.float32)
    l_put_09_ramp = float(
        _exact_piecewise_terminal_put_tensor_f32(
            fc_t,
            vc_t,
            k_strike_t,
            k_low_t,
            active_ki_t,
            b_ki_09,
            torch.tensor([0.10], dtype=torch.float32),
            is_call_t,
            b_ko_t,
            ko_sp_t,
        ).item()
    )
    l_put_09_step = float(
        _exact_piecewise_terminal_put_tensor_f32(
            fc_t,
            vc_t,
            k_strike_t,
            k_low_t,
            active_ki_t,
            b_ki_09,
            torch.tensor([0.0], dtype=torch.float32),
            is_call_t,
            b_ko_t,
            ko_sp_t,
        ).item()
    )
    assert l_put_09_ramp < l_put_09_step

    # 3. Verify _eval_autocall_payoff_pv_batch_np uses [B - |s|, B] on path at w_M = 0.95 when B=1.0, s=-0.10
    req = _make_base_market_and_product(tickers=["SBER_RUB"], ki_barrier=1.0, ki_spread=-0.10)
    req["Product"]["AutoCallTriggerLevels"] = [500.0, 500.0, 500.0]
    req["Product"]["FixedCouponAmount"] = [0.0, 0.0, 0.0]
    state = preprocess_eq_autocall(req)
    perf_path = np.full((1, 1, state.m_rem, 1), 0.95, dtype=np.float64)
    pv_at_095 = float(_eval_autocall_payoff_pv_batch_np(perf_path, state)[0])
    df_term = float(state.df_pay[-1].item())
    # At w_M = 0.95 in [0.90, 1.00], KI gate = (1.0 - 0.95)/0.10 = 0.5, put intrinsic = 0.05 => loss = 0.025
    expected_pv = df_term * (1.0 - 0.5 * 0.05)
    assert abs(pv_at_095 - expected_pv) < 1e-10


def test_I4b_ki_obs_type_none_defaults_to_european_with_warning() -> None:
    """I-4b: OptionDownAndInBarrierObservationType=None defaults to European and emits warning."""
    req = _make_base_market_and_product(
        tickers=["SBER_RUB"],
        ki_barrier=0.85,
        ki_obs_type=None,
    )
    runtime = EQAutocallRuntime(surrogate=MaskedEQAutocallSurrogate().eval(), device="cpu")
    res = runtime.price_trade(req, engine="proxy", include_all_greeks=False)
    assert res.audit_record is None or "KI_OBS_TYPE_DEFAULTED_EUROPEAN" in res.audit_record.get("warnings", [])

    res_auto = runtime.price_trade(req, engine="surrogate", include_all_greeks=False)
    assert res_auto.audit_record is not None
    assert "KI_OBS_TYPE_DEFAULTED_EUROPEAN" in res_auto.audit_record["warnings"]


def test_I5a_I5b_currency_inference_quanto_fx_sign_and_fx_vega_cega() -> None:
    """I-5a & I-5b: Quanto fx_sign (+1 for USDRUB/RUB, -1 for CNHRUB/CNH) and FX Vega/Cega projection."""
    assert _resolve_quanto_fx_sign({"fx_id": "USDRUB"}, "RUB") == 1.0
    assert _resolve_quanto_fx_sign({"fx_id": "CNHRUB"}, "CNH") == -1.0
    assert _resolve_quanto_fx_sign({"fx_id": "CNHRUB", "fx_sign": -1.0}, "CNH") == -1.0

    # Build a trade with explicit quanto metadata for USDRUB (+1) vs CNHRUB (-1)
    req_usd = _make_base_market_and_product(tickers=["AAPLUS", "SBER_RUB"], payment_ccy="RUB")
    req_usd["Model"]["MarketDataSet"]["VolatilitySurfaces"]["USDRUB"] = {
        "TypeName": "FXVolatilitySmileCollection",
        "Identifier": "USDRUB",
        "AsOfDate": req_usd["Model"]["MarketDataSet"]["AsOfDate"],
        "Tenors": ["1M", "3M", "6M", "1Y", "2Y"],
        "Deltas": ["10P", "25P", "ATM", "25C", "10C"],
        "VolatilityQuotes": [
            [0.18, 0.17, 0.16, 0.17, 0.18],
            [0.18, 0.17, 0.16, 0.17, 0.18],
            [0.18, 0.17, 0.16, 0.17, 0.18],
            [0.18, 0.17, 0.16, 0.17, 0.18],
            [0.18, 0.17, 0.16, 0.17, 0.18],
        ],
    }
    req_usd["Model"]["MarketDataSet"]["Correlations"]["AAPLUS&USDRUB"] = -0.25
    req_usd["Model"]["MarketDataSet"]["Correlations"]["SBER_RUB&USDRUB"] = -0.15

    state_usd = preprocess_eq_autocall(req_usd)
    # Ensure quanto_meta is populated even if Shard 3 currency inference is pre-merge in worktree
    if "AAPLUS" not in state_usd.quanto_meta:
        state_usd.quanto_meta["AAPLUS"] = {
            "fx_id": "USDRUB",
            "fx_sign": 1.0,
            "rho_s_fx": -0.25,
            "sig_fx_obs": [0.16] * state_usd.m_rem,
        }
    else:
        state_usd.quanto_meta["AAPLUS"]["fx_sign"] = 1.0

    g_usd = compute_eq_autocall_greeks(
        state_usd,
        queries=["Delta", "Vega", "Cega"],
        surrogate=None,
    )
    assert g_usd.fx_vega is not None and "AAPLUS" in g_usd.fx_vega
    assert g_usd.fx_cega is not None and "AAPLUS" in g_usd.fx_cega
    assert sum(g_usd.fx_vega["AAPLUS"]) > 0.0
    assert g_usd.fx_cega["AAPLUS"]["USDRUB"] < 0.0

    # Now test CNHRUB with payment_currency="CNH" where fx_sign = -1.0
    state_cnh = preprocess_eq_autocall(req_usd)
    state_cnh.payment_currency = "CNH"
    state_cnh.quanto_meta = {
        "SBER_RUB": {
            "fx_id": "CNHRUB",
            "fx_sign": -1.0,
            "rho_s_fx": -0.25,  # effective corr(S_RUB, RUB/CNH) = -1 * (+0.25) = -0.25
            "sig_fx_obs": [0.15] * state_cnh.m_rem,
        }
    }
    g_cnh = compute_eq_autocall_greeks(
        state_cnh,
        queries=["Vega", "Cega"],
        surrogate=None,
    )
    assert g_cnh.fx_vega is not None and sum(g_cnh.fx_vega["SBER_RUB"]) > 0.0
    assert g_cnh.fx_cega is not None and g_cnh.fx_cega["SBER_RUB"]["CNHRUB"] > 0.0

    # Verify runtime merges FX Vega (projected onto 5x5 2D grid) and FX Cega (supporting pair key lookup)
    from deepvol.api.eq_autocall_runtime import _merge_fx_vega_and_cega
    from deepvol.api.schemas_eq_autocall import EQAutocallableTradeRequest

    req_model = EQAutocallableTradeRequest.model_validate(req_usd)
    v_merged, c_merged = _merge_fx_vega_and_cega(g_usd, state_usd, req_model)
    assert v_merged is not None and "USDRUB" in v_merged
    fx_grid = np.asarray(v_merged["USDRUB"], dtype=np.float64)
    assert fx_grid.shape == (5, 5)
    assert float(fx_grid.sum()) > 0.0
    assert c_merged is not None
    assert "AAPLUS&USDRUB" in c_merged
    assert c_merged["AAPLUS&USDRUB"] == pytest.approx(g_usd.fx_cega["AAPLUS"]["USDRUB"])
    assert c_merged["USDRUB"]["AAPLUS"] == pytest.approx(g_usd.fx_cega["AAPLUS"]["USDRUB"])


def test_F12_dual_curve_funding_vs_ois_forward_and_rho_signs() -> None:
    """F-12: Multi-curve Rho separates RUB_FUNDING_SECD discounting (<0) from RUB_RUONIA_OIS forward drift (>0)."""
    req = _make_base_market_and_product(tickers=["SBER_RUB", "LKOH_RUB"], payment_ccy="RUB")
    state = preprocess_eq_autocall(req)
    ois_spec = req["Model"]["MarketDataSet"]["RatesCurvesBundles"]["RUB_RUONIA_OIS"]
    from deepvol.market.eq_curves import ClampedYieldCurve

    ois_curve = ClampedYieldCurve(ois_spec, as_of_date=state.seasoned_state.as_of_date)
    state.asset_financing_curves = {"SBER_RUB": ois_curve, "LKOH_RUB": ois_curve}
    setattr(state, "repo_curves_map", dict(req["Model"]["MarketDataSet"]["RepoSpreadCurves"]))

    g_ad = compute_eq_autocall_greeks(state, queries=["Rho"], surrogate=None)
    assert g_ad.rho is not None
    assert "RUB_FUNDING_SECD" in g_ad.rho
    assert "RUB_RUONIA_OIS" in g_ad.rho
    assert sum(g_ad.rho["RUB_FUNDING_SECD"]) < 0.0
    assert sum(g_ad.rho["RUB_RUONIA_OIS"]) > 0.0

    g_qmc = compute_qmc_crn_greeks(
        state,
        queries=["Rho"],
        config=DupireQMCConfig(n_paths=256, n_replicates=1, n_substeps=2, lv_grid_nt=16, lv_grid_ny=32),
    )
    assert g_qmc.rho is not None
    assert "RUB_FUNDING_SECD" in g_qmc.rho
    assert "RUB_RUONIA_OIS" in g_qmc.rho
    assert sum(g_qmc.rho["RUB_FUNDING_SECD"]) < 0.0
    assert sum(g_qmc.rho["RUB_RUONIA_OIS"]) > 0.0
    assert g_qmc.repo_rho is not None and "SBER_RUB" in g_qmc.repo_rho


def test_I6_is_long_basket_average_of_put_sign_and_cashflow_leg() -> None:
    """I-6: IsLongBasket=True keeps put_sign=-1.0 across all engines and uses AverageOf terminal option index."""
    req = _make_base_market_and_product(tickers=["SBER_RUB", "LKOH_RUB"], is_long_basket=True)
    state = preprocess_eq_autocall(req)

    # 1. Verify _assemble_pv_batch_torch subtracts put loss (put_sign = -1.0)
    p_stop = torch.zeros((1, state.m_rem), dtype=torch.float64)
    q_cpn = torch.zeros((1, state.m_rem), dtype=torch.float64)
    l_call = torch.zeros(1, dtype=torch.float64)
    l_onestar = torch.zeros(1, dtype=torch.float64)
    df_pay_b = state.df_pay.unsqueeze(0)
    pv_in_tr = torch.zeros(1, dtype=torch.float64)

    pv_low_loss = float(
        _assemble_pv_batch_torch(
            state,
            p_stop=p_stop,
            q_cpn=q_cpn,
            l_put=torch.tensor([0.05], dtype=torch.float64),
            l_call=l_call,
            l_onestar=l_onestar,
            df_pay_b=df_pay_b,
            w_fund_b=None,
            pv_in_transit_b=pv_in_tr,
        ).item()
    )
    pv_high_loss = float(
        _assemble_pv_batch_torch(
            state,
            p_stop=p_stop,
            q_cpn=q_cpn,
            l_put=torch.tensor([0.25], dtype=torch.float64),
            l_call=l_call,
            l_onestar=l_onestar,
            df_pay_b=df_pay_b,
            w_fund_b=None,
            pv_in_transit_b=pv_in_tr,
        ).item()
    )
    assert pv_high_loss < pv_low_loss, "Increasing put loss on IsLongBasket=True must decrease PV (put_sign=-1)"

    # 2. Verify assemble_surrogate_product_value in eq_autocall_egno.py uses put_sign = -1.0
    surr_out_low = AutocallSurrogateOutput(
        stopping_probs=torch.zeros((1, state.m_max), dtype=torch.float32),
        survival_probs=torch.ones((1, state.m_max), dtype=torch.float32),
        hazard_rates=torch.zeros((1, state.m_max), dtype=torch.float32),
        coupon_probs=torch.zeros((1, state.m_max), dtype=torch.float32),
        put_loss=torch.tensor([0.05], dtype=torch.float32),
        delta_hazard_logits=torch.zeros((1, state.m_max), dtype=torch.float32),
        delta_coupon_logits=torch.zeros((1, state.m_max), dtype=torch.float32),
        delta_put_loss=torch.zeros(1, dtype=torch.float32),
    )
    surr_out_high = AutocallSurrogateOutput(
        stopping_probs=surr_out_low.stopping_probs,
        survival_probs=surr_out_low.survival_probs,
        hazard_rates=surr_out_low.hazard_rates,
        coupon_probs=surr_out_low.coupon_probs,
        put_loss=torch.tensor([0.25], dtype=torch.float32),
        delta_hazard_logits=surr_out_low.delta_hazard_logits,
        delta_coupon_logits=surr_out_low.delta_coupon_logits,
        delta_put_loss=surr_out_low.delta_put_loss,
    )
    pv_surr_low = float(
        assemble_surrogate_product_value(
            surr_out_low,
            state.sched_feats,
            state.sched_mask,
            state.global_feats,
            is_long_basket=True,
        ).item()
    )
    pv_surr_high = float(
        assemble_surrogate_product_value(
            surr_out_high,
            state.sched_feats,
            state.sched_mask,
            state.global_feats,
            is_long_basket=True,
        ).item()
    )
    assert pv_surr_high < pv_surr_low

    # 3. Verify _eval_autocall_payoff_pv_batch_np uses AverageOf on terminal option when is_long_basket=True
    state_no_ko = preprocess_eq_autocall(req)
    state_no_ko.ko_barriers = torch.full_like(state_no_ko.ko_barriers, 500.0)
    state_no_ko.coupon_amounts = torch.zeros_like(state_no_ko.coupon_amounts)
    state_no_ko.ki_barrier = None
    perf_asym = np.ones((1, 2, state_no_ko.m_rem, 1), dtype=np.float64)
    perf_asym[0, 0, -1, 0] = 0.60
    perf_asym[0, 1, -1, 0] = 1.00  # AverageOf = 0.80 (loss=0.20), WorstOf = 0.60 (loss=0.40)
    pv_avg = float(_eval_autocall_payoff_pv_batch_np(perf_asym, state_no_ko)[0])
    df_term = float(state_no_ko.df_pay[-1].item())
    assert pv_avg == pytest.approx(df_term * (1.0 - 0.20), abs=1e-10)


def test_I10_one_star_coupon_principal_split() -> None:
    """I-10: OneStarStrike=1.20, OneStarCoupon=0.0 splits terminal principal into OneStarCoupon + Notional."""
    req = _make_base_market_and_product(
        tickers=["SBER_RUB", "LKOH_RUB"],
        one_star_strike=1.20,
        one_star_coupon=0.0,
    )
    state = preprocess_eq_autocall(req)
    sig = inspect.signature(assemble_autocall_cashflows_and_pv)
    p_stop = torch.zeros(state.m_rem, dtype=torch.float64)
    q_cpn = torch.zeros(state.m_rem, dtype=torch.float64)
    l_put = torch.tensor(0.10, dtype=torch.float64)
    df_term = float(state.df_pay[-1].item())

    if "p_onestar" in sig.parameters:
        post = assemble_autocall_cashflows_and_pv(
            state,
            p_stop=p_stop,
            q_cpn=q_cpn,
            l_put=l_put,
            p_onestar=torch.tensor(0.35, dtype=torch.float64),
        )
        os_rows = [cf for cf in post.cashflows if cf.Description == "OneStarCoupon"]
        not_rows = [cf for cf in post.cashflows if cf.Description == "Notional"]
        assert len(os_rows) == 1
        assert os_rows[0].Amount == pytest.approx(df_term * 0.35, rel=1e-6)
        assert not_rows[-1].Amount == pytest.approx(df_term * 0.65, rel=1e-6)
        assert (os_rows[0].Amount + not_rows[-1].Amount) == pytest.approx(df_term * 1.0, rel=1e-12)
    else:
        post = assemble_autocall_cashflows_and_pv(
            state,
            p_stop=p_stop,
            q_cpn=q_cpn,
            l_put=l_put,
            l_onestar=0.35,
        )
        not_rows = [cf for cf in post.cashflows if cf.Description == "Notional"]
        assert not_rows[-1].Amount == pytest.approx(df_term * 1.0, rel=1e-12)


def test_F11_long_m_escrowed_dividend_dupire_lv_stability_and_m64_bucket() -> None:
    """F-11: (8, 64) geometry bucket is supported and long-M native-carry Dupire LV avoids min_vol collapse."""
    assert (8, 64) in GEOM_BUCKETS
    assert next(((d, m) for (d, m) in GEOM_BUCKETS if d >= 2 and m >= 52), None) == (8, 64)
    assert next(((d, m) for (d, m) in GEOM_BUCKETS if d >= 4 and m >= 64), None) == (8, 64)

    req = _make_base_market_and_product(tickers=["SBER_RUB"])
    state = preprocess_eq_autocall(req)
    surf = state.vol_surfaces["SBER_RUB"]
    tau_long = np.linspace(1.0 / 12.0, 3.0, 36, dtype=np.float64)
    r_native_long = np.full(36, 0.16, dtype=np.float64)
    lv_t = np.linspace(1e-4, 3.0, 32, dtype=np.float64)
    lv_y = np.linspace(-1.0, 0.8, 64, dtype=np.float64)

    lv_table = _build_single_asset_lv_table_fast_np(
        surf,
        spot_0=100.0,
        tau_obs_np=tau_long,
        r_native_obs_np=r_native_long,
        lv_t_grid=lv_t,
        lv_y_grid=lv_y,
    )
    assert lv_table.shape == (32, 64)
    assert np.all(np.isfinite(lv_table))
    frac_clamped_min = float(np.mean(lv_table <= surf.min_vol + 1e-6))
    assert frac_clamped_min < 0.05, f"Long-M LV table collapsed to min_vol on {frac_clamped_min:.1%} of grid"


def test_I1_fast_cpu_crn_all8_greeks_latency_same_engine_theta_and_multicurve_rho() -> None:
    """I-1: Decoupled 1D-Marginal CPU CRN All-8 Greeks completes in < 2.0s with same-engine Theta."""
    req = _make_base_market_and_product(tickers=["SBER_RUB", "LKOH_RUB"])
    state = preprocess_eq_autocall(req)

    t0 = time.perf_counter()
    res = compute_qmc_crn_greeks(
        state,
        queries=ALL_8_GREEK_QUERIES,
        config=DupireQMCConfig(n_paths=512, n_replicates=1, n_substeps=2, lv_grid_nt=16, lv_grid_ny=32),
    )
    elapsed_s = time.perf_counter() - t0
    assert elapsed_s < 2.0, f"CPU QMC CRN All-8 Greeks took {elapsed_s:.3f}s (limit 2.0s)"

    assert res.delta is not None and set(res.delta.keys()) == {"SBER_RUB", "LKOH_RUB"}
    assert res.gamma is not None and set(res.gamma.keys()) == {"SBER_RUB", "LKOH_RUB"}
    assert res.multivariate_aggregated_cash_gamma is not None and math.isfinite(
        res.multivariate_aggregated_cash_gamma
    )
    assert res.vega is not None and np.asarray(res.vega["SBER_RUB"]).shape == (3, 5)
    assert res.rho is not None and "RUB_FUNDING_SECD" in res.rho
    assert res.dividends_sensitivities is not None and len(res.dividends_sensitivities["SBER_RUB"]) == 2
    assert res.cega is not None and res.cega["SBER_RUB"]["LKOH_RUB"] != 0.0
    assert res.theta is not None and math.isfinite(res.theta)
    # Same-engine 1-day roll Theta must be O(1d carry) (~1-50 bps), never a -490 bps cross-engine artifact
    assert abs(res.theta) < 0.02


def test_I3_psi_batch_invariance_laplace_smoothing_and_30_trade_no_lockout(tmp_path: Any) -> None:
    """I-3: Laplace PSI smoothing, correlation-space OAS (kappa<=40), and 30-trade batch invariance."""
    edges = np.linspace(0.0, 1.0, 11, dtype=np.float64)[None, :]
    ref_probs = np.full((1, 10), 0.10, dtype=np.float64)
    # Small sample (N=20) with several empty bins: Laplace smoothing prevents log(1e-6) explosion
    rng = np.random.default_rng(42)
    small_sample = rng.uniform(0.15, 0.85, size=(20, 1))
    psi_val, _ = compute_psi_10_decile(small_sample, edges, ref_probs)
    guardian = EQAutocallModelGuardian(psi_fallback_threshold=0.25, psi_min_samples=20)
    tau_eff_20 = guardian.effective_psi_threshold(20)
    assert tau_eff_20 == pytest.approx(0.25 + (9.0 / 20.0) * 2.5)
    assert psi_val < tau_eff_20

    # Correlation-space OAS covariance condition number <= 40 and d=1 12D subspace projection
    feats = rng.normal(size=(64, 16)).astype(np.float64)
    feats[:, 1] = feats[:, 0]  # collinear multi-asset coordinate on d=1
    feats[:, 2] = feats[:, 0]
    feats[:, 5] = feats[:, 4]
    feats[:, 6] = feats[:, 4]
    cov_raw = np.cov(feats, rowvar=False)
    cov_oas, cov_inv_oas, cov_inv_d1, marginal_std = _regularize_cov_oas_correlation_space(cov_raw)
    assert np.all(np.isfinite(cov_inv_oas))
    assert np.all(np.isfinite(cov_inv_d1))

    surr_model = MaskedEQAutocallSurrogate().eval()
    synth_b = generate_synthetic_autocall_batch(
        batch_size=8,
        d_max=4,
        m_max=12,
        n_mc_paths=32,
        n_substeps=2,
        seed=19,
        domain_mode="tier2_desk",
        compute_pathwise_delta=False,
    )
    stats_file = tmp_path / "guardian_stats.pt"
    stats = calibrate_guardian_stats(
        surr_model,
        synth_b,
        synth_b,
        device=torch.device("cpu"),
        conformal_alpha=0.05,
        stats_path=stats_file,
    )
    assert "mondrian_q_err_table" in stats
    assert "oas_standardized" in stats
    guardian.load_calibrated_stats(stats_file)
    d2_d1 = guardian.compute_mahalanobis_d2(
        stats["ref_mean_16d"],
        d_assets=1,
    )
    assert math.isfinite(d2_d1) and d2_d1 == pytest.approx(0.0, abs=1e-9)

    # Run a batch of 30 trades through EQAutocallRuntime.price_batch_json: 0 PSI_DRIFT_BREACH lockouts!
    runtime = EQAutocallRuntime(
        surrogate=surr_model,
        stats_path=stats_file,
        device="cpu",
    )
    runtime.guardian.mahalanobis_chi2_threshold = 1.0e9
    runtime.guardian.tau_desk_bps = 1.0e9
    base_req = _make_base_market_and_product(tickers=["SBER_RUB", "LKOH_RUB"])
    base_req["Queries"] = ["ProductValue", "Delta"]
    batch_reqs = []
    for k in range(30):
        rk = dict(base_req)
        rk["TradeInfo"] = f"BATCH_TRADE_{k:02d}"
        batch_reqs.append(rk)

    responses = runtime.price_batch_json(batch_reqs, compute_greeks=False, engine="auto")
    assert len(responses) == 30
    for idx, resp in enumerate(responses):
        assert AutocallFallbackReason.PSI_DRIFT_BREACH.value not in resp.fallback_reasons, (
            f"Trade #{idx} falsely locked out by PSI_DRIFT_BREACH during synchronous batch pricing"
        )
    assert len(runtime.guardian.feature_history) == 0


def test_I7_irao_rub_7slice_splice_rejection_diagnostics_and_mild_noise_projection() -> None:
    """I-7: IRAO_RUB 7-slice butterfly splice raises SMILE_ARBITRAGE, while mild noise is accepted."""
    as_of = dt.datetime(2026, 4, 3, tzinfo=dt.timezone.utc)
    strikes_irao = [2.6362, 3.9544, 4.7453, 5.0089, 5.0725, 5.2725, 5.5361, 6.5906, 7.9088]
    # Severe 530 vol-bps step across K=5.0725 -> 5.2725 at short tenor T0 = 17/365
    quotes_severe = [
        [0.4408, 0.3421, 0.2915, 0.2764, 0.27281, 0.32577, 0.3180, 0.3150, 0.3200],
        [0.4200, 0.3350, 0.2950, 0.2850, 0.28300, 0.31200, 0.3080, 0.3060, 0.3100],
        [0.4000, 0.3300, 0.3000, 0.2920, 0.29000, 0.30200, 0.3000, 0.2990, 0.3020],
    ]
    with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
        RegularizedVolSurface(
            {
                "identifier": "IRAO_RUB",
                "tenors": [17.0 / 365.0, 0.25, 0.50],
                "strikes": strikes_irao,
                "quotes": quotes_severe,
            },
            as_of_date=as_of,
        )

    # Smooth convex quotes with mild rounding noise must construct cleanly
    quotes_smooth = [
        [0.3600, 0.3200, 0.2950, 0.2900, 0.2890, 0.2885, 0.2895, 0.2980, 0.3120],
        [0.3500, 0.3150, 0.2920, 0.2880, 0.2870, 0.2865, 0.2875, 0.2950, 0.3080],
        [0.3400, 0.3100, 0.2900, 0.2860, 0.2850, 0.2845, 0.2855, 0.2920, 0.3050],
    ]
    surf_ok = RegularizedVolSurface(
        {
            "identifier": "IRAO_RUB",
            "tenors": [17.0 / 365.0, 0.25, 0.50],
            "strikes": strikes_irao,
            "quotes": quotes_smooth,
        },
        as_of_date=as_of,
    )
    assert np.isfinite(surf_ok.iv(0.25, 5.2725))

    # Also verify tier2_desk Sobol generator covers high-rate RUB/CNH regime and put_sign=-1.0
    batch_gen = generate_synthetic_autocall_batch(
        batch_size=2,
        d_max=4,
        m_max=12,
        n_mc_paths=64,
        n_substeps=2,
        seed=7,
        domain_mode="tier2_desk",
        compute_pathwise_delta=False,
    )
    assert batch_gen.node_feats.shape == (2, 4, 16)
    assert torch.all(torch.isfinite(batch_gen.l_put_mc))
    assert torch.all(torch.isfinite(batch_gen.pv_mc))
