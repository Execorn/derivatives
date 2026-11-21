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
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


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


def test_m3_1d_vanilla_put_with_cash_dividends_matches_black76(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    import datetime
    import math
    from scipy.special import ndtr

    req_orig, _ = parsed_dag
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    t_mat = as_of + datetime.timedelta(days=365)
    sigma = 0.25
    r_zero = 0.05
    strike = 1.00
    s0 = 100.0

    for div_amt in (0.0, 10.0, 20.0, 30.0):
        req = copy.deepcopy(req_orig)
        req.Product.StartDate = as_of
        req.Product.ObservationDates = [t_mat]
        req.Product.PaymentDates = [t_mat]
        req.Product.AutoCallTriggerLevels = [500.0]
        req.Product.CouponBarrierLevels = [500.0]
        req.Product.FixedCouponAmount = 0.0
        req.Product.FixedRate = None
        req.Product.IsMemory = False
        req.Product.Strike = strike
        req.Product.OptionGearing = 1.0
        req.Product.OptionDownAndInBarrierLevel = None
        req.Product.NoNotionalRepayment = True
        req.Product.AutoCallTriggerAbsoluteDigitalSpread = 0.0
        req.Product.UnderlyingIdentifiers = ["EQ0_RUB"]
        req.Product.UnderlyingReferenceSpots = [s0]
        req.Product.PredefinedDividends = (
            {
                "EQ0_RUB": {
                    "ExDates": [as_of + datetime.timedelta(days=182)],
                    "Amounts": [div_amt],
                }
            }
            if div_amt > 0.0
            else None
        )
        mds = req.Model.MarketDataSet
        mds.AsOfDate = as_of
        mds.Spots = {"EQ0_RUB": s0}
        mds.Correlations = {}
        mds.Dividends = {}
        mds.Fixings = {}
        mds.RepoSpreadCurves = {}
        from deepvol.api.schemas_eq_autocall import CurveInterpolationMethodEnum
        disc = mds.RatesCurvesBundles["RUB_RUONIA_OIS"]
        disc.AsOfDate = as_of
        disc.InterpolationMethod = CurveInterpolationMethodEnum.Linear
        disc.RatesSchedule.Dates = [as_of + datetime.timedelta(days=90), t_mat, as_of + datetime.timedelta(days=730)]
        disc.RatesSchedule.Rates = [r_zero * 100.0] * 3
        mds.VolatilitySurfaces = {
            "EQ0_RUB": {
                "Identifier": "EQ0_RUB",
                "DayCountConvention": "Act365",
                "ExpirationDates": [
                    (as_of + datetime.timedelta(days=90)).isoformat(),
                    t_mat.isoformat(),
                    (as_of + datetime.timedelta(days=730)).isoformat(),
                ],
                "Strikes": [50.0, 80.0, 100.0, 120.0, 150.0],
                "VolatilityQuotes": [[sigma] * 5 for _ in range(3)],
            }
        }

        state = preprocess_eq_autocall(req)
        res = price_eq_autocall_dupire_qmc(state, n_paths=16384, n_replicates=8, mode="dupire_lv", n_substeps_per_period=8)
        f_perf = float(state.forward_perf[0, 0].item())
        df = float(state.df_pay[0].item())
        v = sigma * math.sqrt(float(state.tau_obs[0].item()))
        d1 = (math.log(f_perf / strike) + 0.5 * v * v) / v
        d2 = d1 - v
        b76_pv = -df * (strike * float(ndtr(-d2)) - f_perf * float(ndtr(-d1)))
        assert abs(res.product_value - b76_pv) <= 5e-5, (
            f"div={div_amt}: QMC={res.product_value:.6f} vs B76={b76_pv:.6f}"
        )


def test_m3_durrleman_butterfly_immutability_high_dim_and_exotic_qmc(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    import datetime
    from deepvol.models.autocall.dupire_qmc_solver import _generate_sobol_normals

    req_orig, _ = parsed_dag
    as_of = req_orig.Model.MarketDataSet.AsOfDate
    state = preprocess_eq_autocall(req_orig)

    # CHR-010: RegularizedVolSurface arrays are immutable
    for surf in state.vol_surfaces.values():
        assert surf.strikes.flags.writeable is False
        assert surf.quotes.flags.writeable is False
        assert surf.tenors.flags.writeable is False
        iv_b = surf.iv_batch(np.array([0.2, 0.5], dtype=np.float64), np.array([surf.strikes[0], surf.strikes[-1]], dtype=np.float64))
        assert iv_b.shape == (2,) and np.all(np.isfinite(iv_b))
        assert surf.calendar_proj_digest is not None

    # SCH-015 & T1M-012: Linear/Flat smile options & Act360 surface DCC
    vkco_spec = copy.deepcopy(req_orig.Model.MarketDataSet.VolatilitySurfaces["VKCO_RUB"])
    vkco_spec.DayCountConvention = "Act360"
    vkco_surf_lin = RegularizedVolSurface(vkco_spec, as_of_date=as_of, smile_interp="Linear", smile_extrap="Flat")
    assert np.isfinite(float(vkco_surf_lin.iv(0.3, 600.0)))

    # EXT-005: Violent sawtooth butterfly arbitrage raises SMILE_ARBITRAGE
    saw_tenors = np.array([0.5, 1.0], dtype=np.float64)
    saw_strikes = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0], dtype=np.float64)
    saw_quotes = np.array(
        [
            [0.20, 0.95, 0.12, 0.95, 0.12, 0.95, 0.20],
            [0.20, 0.95, 0.12, 0.95, 0.12, 0.95, 0.20],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
        enforce_no_arbitrage_projection(saw_tenors, saw_strikes, saw_quotes)

    # T1M-009: High-dim Sobol (> 21201) fallback to PCG64 antithetic without crash
    z_hi = _generate_sobol_normals(n_paths=64, dim=21220, scramble=True, seed=7)
    assert z_hi.shape == (64, 21220)
    assert np.all(np.isfinite(z_hi))

    # T1M-002 & SCH-004 & SCH-016: 6-period memory Phoenix + RatioAB + OneStar + PartCall + Out/Down + FundingLeg
    req_multi = copy.deepcopy(req_orig)
    as_of_m = datetime.datetime(2026, 6, 1, 0, 0, tzinfo=datetime.timezone.utc)
    req_multi.Model.MarketDataSet.AsOfDate = as_of_m
    for c in req_multi.Model.MarketDataSet.RatesCurvesBundles.values():
        c.AsOfDate = as_of_m
    req_multi.Product.UnderlyingIdentifiers = ["VKCO_RUB", "MAGN_RUB"]
    req_multi.Product.UnderlyingReferenceSpots = [661.0, 58.4]
    req_multi.Product.CallabilityMonitoringIndex = "RatioAB"
    req_multi.Product.CouponBarrierMonitoringIndex = "DiffAB"
    req_multi.Product.OptionMonitoringIndex = "WorstOf"
    req_multi.Product.CouponBarrierType = "Out"
    req_multi.Product.CouponBarrierDirection = "Down"
    req_multi.Product.CouponBarrierLevels = [0.75] * 12
    req_multi.Product.LowerStrike = 0.50
    req_multi.Product.AllowAutocallOnExpiry = True
    req_multi.Product.AutoCallTriggerLevels[-1] = 0.95
    req_multi.Product.AutoCallCoupon = 0.03
    req_multi.Product.ForceCouponsOnAutocall = True
    req_multi.Product.OneStarStrike = 1.10
    req_multi.Product.OneStarCoupon = 0.04
    req_multi.Product.ParticipationCallStrike = 1.00
    req_multi.Product.ParticipationCallRate = 0.25
    req_multi.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_multi.Product.FundingSpread = 0.02
    req_multi.Product.FundingLegPaymentDates = list(req_multi.Product.PaymentDates)

    st_multi = preprocess_eq_autocall(req_multi)
    res_m = price_eq_autocall_dupire_qmc(
        st_multi,
        n_paths=1024,
        n_replicates=4,
        mode="dupire_lv",
        n_substeps_per_period=2,
    )
    assert np.isfinite(res_m.product_value)
    assert res_m.standard_error > 0.0
    assert len(res_m.cashflows) > 0


def test_m3_wing_projection_surface_dcc_and_validation_errors() -> None:
    import datetime
    import math
    from deepvol.models.autocall.dupire_qmc_solver import _invert_b76_norm_call_to_vol, _surface_year_fraction

    # _surface_year_fraction aliases
    d1 = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    d2 = datetime.datetime(2026, 7, 2, 0, 0, tzinfo=datetime.timezone.utc)
    for dcc_alias in ("Act365", "Act365_25", "Act360", "Act364", "Act366_1", "Act252", "Business252", "ActActISDA", "30E360", "UnknownCustom"):
        yf = _surface_year_fraction(d1, d2, dcc_alias)
        assert 0.4 < yf < 0.8

    # Wing convexity projection + _invert_b76_norm_call_to_vol
    wing_t = np.array([0.5, 1.0], dtype=np.float64)
    wing_k = np.array([30.0, 60.0, 90.0, 100.0, 110.0, 140.0, 180.0], dtype=np.float64)
    wing_q = np.array(
        [
            [0.08, 0.25, 0.22, 0.20, 0.22, 0.25, 0.45],
            [0.26, 0.27, 0.24, 0.22, 0.24, 0.27, 0.46],
        ],
        dtype=np.float64,
    )
    _, q_proj, _, changed = enforce_no_arbitrage_projection(wing_t, wing_k, wing_q)
    assert changed is True and np.all(np.isfinite(q_proj))
    assert _invert_b76_norm_call_to_vol(0.0, 0.0, 1.0, 0.01, 5.0) == pytest.approx(0.01)
    assert _invert_b76_norm_call_to_vol(0.0, 1.0, 1.0, 0.01, 5.0) == pytest.approx(5.0)

    # RegularizedVolSurface Mapping spec with tenors & validation errors
    surf_dict = RegularizedVolSurface(
        {
            "Identifier": "SYN_SURF",
            "DayCountConvention": "Act365",
            "tenors": [0.25, 1.0],
            "Strikes": [80.0, 100.0, 120.0],
            "VolatilityQuotes": [[0.25, 0.22, 0.25], [0.24, 0.21, 0.24]],
        }
    )
    assert np.isfinite(float(surf_dict.iv(0.5, 100.0)))
    assert len(surf_dict.iv_batch(np.array([], dtype=np.float64), np.array([], dtype=np.float64))) == 0

    with pytest.raises(ValueError):
        RegularizedVolSurface({"tenors": [0.5], "strikes": [80.0, 100.0], "quotes": [0.2, 0.2]})
    with pytest.raises(ValueError):
        RegularizedVolSurface({"tenors": [1.0, 0.5], "strikes": [80.0, 100.0], "quotes": [[0.2, 0.2], [0.2, 0.2]]})
    surf_rev = RegularizedVolSurface({"tenors": [0.5, 1.0], "strikes": [100.0, 80.0], "quotes": [[0.2, 0.25], [0.21, 0.26]]})
    assert list(surf_rev.strikes) == pytest.approx([80.0, 100.0])
    with pytest.raises(ValueError):
        RegularizedVolSurface({"tenors": [0.5, 1.0], "strikes": [80.0, 100.0], "quotes": [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]})

    # natural bc, iv_and_d_iv_d_quotes, scalar/vec local_vol, mode="black76", and n_replicates=3 warning
    surf_nat = RegularizedVolSurface(
        {
            "Identifier": "NAT_SURF",
            "DayCountConvention": "Act365",
            "tenors": [0.25, 1.0, 2.0],
            "Strikes": [75.0, 90.0, 105.0, 120.0],
            "VolatilityQuotes": [[0.25, 0.23, 0.22, 0.24], [0.24, 0.22, 0.21, 0.23], [0.23, 0.21, 0.20, 0.22]],
        },
        bc_type="natural",
    )
    iv_v, d_iv_dq = surf_nat.iv_and_d_iv_d_quotes(0.5, 100.0)
    assert np.isfinite(iv_v) and d_iv_dq.shape == (3, 4)
    # Non-square n_exp != n_strikes - 1 regression check for CubicSpline axis layout (#76)
    surf_3x3 = RegularizedVolSurface(
        {
            "Identifier": "NAT_3X3",
            "DayCountConvention": "Act365",
            "tenors": [0.25, 1.0, 2.0],
            "Strikes": [80.0, 100.0, 120.0],
            "VolatilityQuotes": [[0.25, 0.22, 0.24], [0.24, 0.21, 0.23], [0.23, 0.20, 0.22]],
        },
        bc_type="natural",
    )
    assert math.isfinite(float(surf_3x3.local_vol(0.5, 100.0)))
    for t_v in (0.05, 0.5, 3.0):
        lv_sc = surf_nat.local_vol(t_v, np.array([70.0, 100.0, 130.0], dtype=np.float64))
        assert lv_sc.shape == (3,) and np.all(np.isfinite(lv_sc))
    assert math.isfinite(float(surf_nat.local_vol(0.5, 100.0)))

    # Exercise price_eq_autocall_dupire_qmc mode="terminal_iv" and non-pow2 replicate warning
    req_orig, _ = parse_murex_autocall_dag(
        Path(__file__).resolve().parents[1] / "data" / "autocall" / "eq_autocallable_request.txt"
        if (Path(__file__).resolve().parents[1] / "data" / "autocall" / "eq_autocallable_request.txt").exists()
        else Path("/home/execorn/programming/derivatives/data/autocall/eq_autocallable_request.txt")
    )
    st_q = preprocess_eq_autocall(req_orig)
    res_tiv = price_eq_autocall_dupire_qmc(st_q, n_paths=128, n_replicates=3, mode="terminal_iv")
    assert np.isfinite(res_tiv.product_value)
    res_raw = price_eq_autocall_dupire_qmc(req_orig, n_paths=64, n_replicates=2, mode="auto")
    assert np.isfinite(res_raw.product_value)

    # Strike basis weights across Linear, NaturalCubicSpline, NotAKnot, Linear/Flat extrap + Act360 tenor mapping
    for interp_m, extrap_m, bc_m in (
        ("Linear", "Linear", "clamped"),
        ("Linear", "Flat", "clamped"),
        ("CubicSpline", "Linear", "clamped"),
        ("CubicSpline", "Flat", "clamped"),
        ("NaturalCubicSpline", "Linear", "natural"),
        ("NotAKnot", "Flat", "not-a-knot"),
    ):
        surf_b = RegularizedVolSurface(
            {
                "Identifier": f"BASIS_{interp_m}_{extrap_m}",
                "DayCountConvention": "Act360",
                "tenors": [0.25, 1.0, 2.0],
                "Strikes": [80.0, 100.0, 120.0],
                "VolatilityQuotes": [[0.25, 0.22, 0.24], [0.24, 0.21, 0.23], [0.23, 0.20, 0.22]],
            },
            bc_type=bc_m,
            smile_interp=interp_m,
            smile_extrap=extrap_m,
        )
        w_vec = surf_b.strike_basis_weights(np.array([65.0, 100.0, 135.0], dtype=np.float64))
        assert w_vec.shape == (3, 3) and np.allclose(w_vec.sum(axis=-1), 1.0, atol=1e-10)
        assert surf_b.map_act365_to_surface_tenor(0.5) > 0.0
        assert surf_b.map_act365_to_surface_tenor(5.0) > 2.0

    with pytest.raises(ValueError, match="as_of_date"):
        surf_nat.tenor_from_date(datetime.datetime(2027, 1, 1, tzinfo=datetime.timezone.utc))

    # Exercise min_vol clamping during linear extrapolation, scramble=False, and already_autocalled
    surf_lin_neg = RegularizedVolSurface(
        {
            "Identifier": "LIN_CLAMP",
            "DayCountConvention": "Act365",
            "tenors": [0.25, 1.0],
            "Strikes": [90.0, 100.0, 110.0],
            "VolatilityQuotes": [[0.20, 0.25, 0.30], [0.22, 0.27, 0.32]],
        },
        smile_interp="Linear",
        smile_extrap="Linear",
    )
    assert float(surf_lin_neg.local_vol(0.5, 1.0)) >= 0.0
    res_unscr = price_eq_autocall_dupire_qmc(st_q, n_paths=64, scramble=False, mode="terminal_iv")
    assert np.isfinite(res_unscr.product_value)
    st_m0 = copy.copy(st_q)
    st_m0.m_rem = 0
    res_m0 = price_eq_autocall_dupire_qmc(st_m0, n_paths=64)
    assert res_m0.standard_error == 0.0





