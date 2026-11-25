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


def test_dupire_qmc_audit_blind_spots(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    import datetime
    from deepvol.models.autocall.dupire_qmc_solver import enforce_no_arbitrage_projection_with_jacobian
    from deepvol.models.autocall.feature_layout import MonitoringIndexCode

    # Scale-invariant Durrleman convexity projection at tiny K~0.01 and huge K~50000.
    for scale in (1e-4, 500.0):
        strikes_s = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=np.float64) * scale
        tenors_s = np.array([0.25, 1.0], dtype=np.float64)
        q_raw = np.array(
            [
                [0.30, 0.25, 0.27, 0.24, 0.30],
                [0.29, 0.25, 0.26, 0.25, 0.29],
            ],
            dtype=np.float64,
        )
        _, proj_q, _, _, _, _, jac_p = enforce_no_arbitrage_projection_with_jacobian(
            tenors_s,
            strikes_s,
            q_raw,
        )
        assert proj_q.shape == q_raw.shape and np.all(np.isfinite(proj_q))
        assert jac_p.shape[0] == 2 and np.all(np.isfinite(jac_p))
        surf_s = RegularizedVolSurface(
            {
                "Identifier": f"SCALE_{scale}",
                "DayCountConvention": "Act365",
                "tenors": [0.25, 1.0],
                "Strikes": strikes_s.tolist(),
                "VolatilityQuotes": proj_q.tolist(),
            },
            as_of_date=datetime.datetime(2026, 10, 23, tzinfo=datetime.timezone.utc),
        )
        assert surf_s.iv(0.5, 100.0 * scale) > 0.0

    # Capitalized "Tenors" numeric list must parse cleanly alongside lowercase "tenors".
    surf_cap_t = RegularizedVolSurface(
        {
            "Identifier": "TENORS_CAP",
            "DayCountConvention": "Act365",
            "Tenors": [0.25, 1.0],
            "Strikes": [80.0, 100.0, 120.0],
            "VolatilityQuotes": [[0.25, 0.22, 0.24], [0.24, 0.21, 0.23]],
        },
        as_of_date=datetime.datetime(2026, 10, 23, tzinfo=datetime.timezone.utc),
    )
    assert surf_cap_t.iv(0.5, 100.0) > 0.0

    # All 18 indices including SecondWorstOf (15), ThirdWorstOf (16), SecondBestOf (17),
    # EquallyWeightedBasket, MultiPerf, and d==1 pair-index edge cases.
    perf4 = np.array([[0.82, 0.94, 1.07, 1.22], [1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    perf4_t = torch.tensor(perf4, dtype=torch.float64)
    for idx_name, expected_row0 in (
        ("SecondWorstOf", 0.94),
        ("ThirdWorstOf", 1.07),
        ("SecondBestOf", 1.07),
        ("EquallyWeightedBasket", float(np.mean(perf4[0]))),
        ("MultiPerf", float(np.mean(perf4[0]))),
    ):
        v_np = evaluate_monitoring_index_np(perf4, idx_name, None)
        v_th = evaluate_monitoring_index_torch(perf4_t, idx_name, None)
        assert float(v_np[0]) == pytest.approx(expected_row0, abs=1e-12)
        np.testing.assert_allclose(v_np, v_th.numpy(), atol=1e-12)
        assert int(MonitoringIndexCode.SECOND_WORST_OF) == 15

    p1 = np.array([[1.08]], dtype=np.float64)
    p1_t = torch.tensor(p1, dtype=torch.float64)
    assert float(evaluate_monitoring_index_np(p1, "RatioAB", None)[0]) == pytest.approx(1.08, abs=1e-12)
    assert float(evaluate_monitoring_index_torch(p1_t, "RatioAB", None)[0]) == pytest.approx(1.08, abs=1e-12)
    assert float(evaluate_monitoring_index_np(p1, "DiffAB", None)[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(evaluate_monitoring_index_torch(p1_t, "DiffAB", None)[0]) == pytest.approx(0.0, abs=1e-12)

    # CPU solver: cv_mode="none" vs "bs1d_exact", one_star_coupon < 0, Continuous KI vs European,
    # and In-Transit FundingLeg alpha_rem + funding_fixings matching preprocessor.
    req_orig, _ = parsed_dag
    req_fl = copy.deepcopy(req_orig)
    req_fl.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_fl.Product.FundingSpread = 0.02
    req_fl.Product.FundingLegPaymentDates = list(req_fl.Product.PaymentDates)
    as_of_fl = req_fl.Model.MarketDataSet.AsOfDate
    req_fl.Product.PaymentDates[-2] = as_of_fl + datetime.timedelta(days=10)
    req_fl.Product.FundingLegPaymentDates[-2] = as_of_fl + datetime.timedelta(days=10)
    st_fl = preprocess_eq_autocall(req_fl)
    res_fl_none = price_eq_autocall_dupire_qmc(st_fl, n_paths=2048, n_replicates=4, cv_mode="none", seed=42)
    res_fl_cv = price_eq_autocall_dupire_qmc(st_fl, n_paths=2048, n_replicates=4, cv_mode="bs1d_exact", seed=42)
    assert abs(res_fl_none.product_value - res_fl_cv.product_value) < 0.015

    # Continuous KI increases downside loss relative to European KI on deep barrier breach paths.
    req_ki = copy.deepcopy(req_orig)
    req_ki.Model.MarketDataSet.AsOfDate = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    for c in req_ki.Model.MarketDataSet.RatesCurvesBundles.values():
        c.AsOfDate = req_ki.Model.MarketDataSet.AsOfDate
    req_ki.Product.OptionDownAndInBarrierLevel = 0.85
    req_ki.Product.OptionDownAndInBarrierObservationType = "European"
    st_eu = preprocess_eq_autocall(req_ki)
    req_ki.Product.OptionDownAndInBarrierObservationType = "Continuous"
    st_cont = preprocess_eq_autocall(req_ki)
    pv_eu = price_eq_autocall_dupire_qmc(st_eu, n_paths=2048, n_replicates=4, seed=11)
    pv_cont = price_eq_autocall_dupire_qmc(st_cont, n_paths=2048, n_replicates=4, seed=11)
    assert np.isfinite(pv_eu.product_value) and np.isfinite(pv_cont.product_value)
    assert pv_cont.l_put >= pv_eu.l_put - 1e-4

    # Negative one_star_coupon (!= 0) must affect option PV cleanly.
    req_os = copy.deepcopy(req_ki)
    req_os.Product.OneStarStrike = 1.05
    req_os.Product.OneStarCoupon = -0.04
    res_os_neg = price_eq_autocall_dupire_qmc(preprocess_eq_autocall(req_os), n_paths=512, n_replicates=2, seed=7)
    req_os.Product.OneStarCoupon = 0.0
    res_os_zero = price_eq_autocall_dupire_qmc(preprocess_eq_autocall(req_os), n_paths=512, n_replicates=2, seed=7)
    assert abs(res_os_neg.product_value - res_os_zero.product_value) > 1e-5


def test_v3_literal_memory_and_pure_sobol_bb_determinism(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify V3 literal survival-weighted unpaid_mem update and deterministic pure Sobol BB."""
    from types import SimpleNamespace
    from deepvol.models.autocall.dupire_qmc_solver import _digital_ramp_np, _evaluate_autocall_payoff_np

    # Deterministic 2-period pathwise test comparing against exact V3 literal algebra:
    # Period 0: W=[1.00, 1.00], ko_bar=1.00, ko_spread=-0.10 -> ko_ind=1.0? No, choose W=0.95 -> ko_ind=0.5, cpn_ind=0.5!
    # Period 1: W=1.05 -> ko_ind=1.0, cpn_ind=1.0.
    st_mock = SimpleNamespace(
        seasoned_state=SimpleNamespace(c_mem_0=0.10, i_ki_past=0.0),
        callability_index="WorstOf",
        coupon_index="WorstOf",
        option_index="WorstOf",
        is_long_basket=False,
        coupon_barrier_direction="Up",
        coupon_barrier_type="In",
        ki_barrier=None,
        ki_spread=0.0,
        ko_spread=-0.10,
        cpn_spread=-0.10,
        force_coupons_on_autocall=False,
        is_memory=True,
        autocall_coupon=0.0,
        no_notional_repayment=False,
        one_star_strike=None,
        one_star_coupon=0.0,
        strike=1.0,
        option_gearing=0.0,
        lower_strike=None,
        participation_call_strike=None,
        participation_call_rate=0.0,
    )
    perf_paths = np.array([[[0.95], [1.05]]], dtype=np.float64)
    df_pay = np.array([0.95, 0.90], dtype=np.float64)
    ko_bar = np.array([1.00, 1.00], dtype=np.float64)
    cpn_bar = np.array([1.00, 1.00], dtype=np.float64)
    is_call = np.array([1.0, 1.0], dtype=np.float64)
    is_guar = np.array([0.0, 0.0], dtype=np.float64)
    cpn_amt = np.array([0.05, 0.05], dtype=np.float64)
    w_fund = np.array([0.0, 0.0], dtype=np.float64)

    _, _, _, _, cpn_cf, *_ = _evaluate_autocall_payoff_np(
        st_mock,
        perf_paths,
        df_pay=df_pay,
        ko_barriers=ko_bar,
        is_callable=is_call,
        cpn_barriers=cpn_bar,
        is_guaranteed_cpn=is_guar,
        coupon_amounts=cpn_amt,
        w_fund=w_fund,
        inv_lex_perm=None,
    )
    # Period 0: alive=1.0, ko_ind=0.5, cpn_ind=0.5 -> cpn_0 = 1.0*0.5*(0.05+0.10)=0.075
    # V3 literal: unpaid_mem = alive * (0.10 + 0.05) * (1 - 0.5) = 0.075 (NOT 0.075/1 without alive, wait alive=1 at m=0;
    # at m=1: alive_1=0.5, cpn_1 = 0.5 * 1.0 * (0.05 + 0.075) = 0.0625)
    # Test 3 periods so alive_1 = 0.5 enters unpaid_mem_1!
    perf3 = np.array([[[0.95], [0.95], [1.05]]], dtype=np.float64)
    df3 = np.array([0.95, 0.90, 0.85], dtype=np.float64)
    ko3 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    ic3 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    ig3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    c3 = np.array([0.05, 0.05, 0.05], dtype=np.float64)
    wf3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    _, _, _, _, cpn_cf3, *_ = _evaluate_autocall_payoff_np(
        st_mock,
        perf3,
        df_pay=df3,
        ko_barriers=ko3,
        is_callable=ic3,
        cpn_barriers=ko3,
        is_guaranteed_cpn=ig3,
        coupon_amounts=c3,
        w_fund=wf3,
        inv_lex_perm=None,
    )
    k0 = float(_digital_ramp_np(np.array([0.95]), 1.0, -0.10)[0])
    assert k0 == pytest.approx(0.5, abs=1e-12)
    # m=0: a0=1.0, cpn0=0.075, U0=1.0*(0.10+0.05)*0.5=0.075, a1=0.5
    # m=1: a1=0.5, cpn1=0.5*0.5*(0.05+0.075)=0.03125, U1=0.5*(0.075+0.05)*0.5=0.03125, a2=0.25
    # m=2: a2=0.25, cpn2=0.25*1.0*(0.05+0.03125)=0.0203125 (vs V1 unweighted: 0.25*(0.05+0.0625)=0.028125)
    assert cpn_cf3[0] == pytest.approx(0.075, abs=1e-12)
    assert cpn_cf3[1] == pytest.approx(0.03125, abs=1e-12)
    assert cpn_cf3[2] == pytest.approx(0.0203125, abs=1e-12)

    # Pure Sobol BB determinism check on parsed_dag
    req_orig, _ = parsed_dag
    st = preprocess_eq_autocall(req_orig)
    r1 = price_eq_autocall_dupire_qmc(st, n_paths=512, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=17)
    r2 = price_eq_autocall_dupire_qmc(st, n_paths=512, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=17)
    assert r1.product_value == r2.product_value
    assert r1.option_pv == r2.option_pv
    assert r1.coupon_pv == r2.coupon_pv


def test_buehler_eff_drift_1d_skewed_div_lv_reprices_smile(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Gate G9 proxy: 1D skewed discrete-dividend Dupire LV reprices 1.0F/1.2F within <= 10 bps IV."""
    import datetime
    import math
    from scipy.special import ndtr
    from deepvol.api.schemas_eq_autocall import CurveInterpolationMethodEnum

    req_orig, _ = parsed_dag
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    t_mat = as_of + datetime.timedelta(days=730)
    s0 = 100.0
    r_zero = 0.14
    div_dates_days = [200, 560]
    div_amts = [3.5, 3.5]
    strikes = [50.0, 65.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 145.0, 165.0, 190.0]
    exp_days = [90, 180, 365, 547, 730, 912]
    # Smooth market smile centered on continuous-yield forward matching 2Y escrowed forward
    q_eff_avg = 0.036
    quotes = []
    for d_exp in exp_days:
        t_e = d_exp / 365.0
        f_e = s0 * math.exp((r_zero - q_eff_avg) * t_e)
        row_q = [max(0.28 - 0.10 * math.log(k_v / f_e) + 0.03 * (math.log(k_v / f_e) ** 2), 0.15) for k_v in strikes]
        quotes.append(row_q)

    req = copy.deepcopy(req_orig)
    req.Product.StartDate = as_of
    req.Product.ObservationDates = [
        as_of + datetime.timedelta(days=91 * k) for k in range(1, 8)
    ] + [t_mat]
    req.Product.PaymentDates = list(req.Product.ObservationDates)
    req.Product.AutoCallTriggerLevels = [999.0] * 8
    req.Product.CouponBarrierLevels = [999.0] * 8
    req.Product.FixedCouponAmount = 0.0
    req.Product.FixedRate = None
    req.Product.IsMemory = False
    req.Product.OptionDownAndInBarrierLevel = None
    req.Product.OneStarStrike = None
    req.Product.OneStarCoupon = None
    req.Product.NoNotionalRepayment = True
    req.Product.OptionGearing = 1.0
    req.Product.OptionMonitoringIndex = "WorstOf"
    req.Product.UnderlyingIdentifiers = ["SKEW_RUB"]
    req.Product.UnderlyingReferenceSpots = [s0]
    req.Product.PredefinedDividends = {
        "SKEW_RUB": {
            "ExDates": [as_of + datetime.timedelta(days=d_ex) for d_ex in div_dates_days],
            "Amounts": div_amts,
        }
    }
    mds = req.Model.MarketDataSet
    mds.AsOfDate = as_of
    mds.Spots = {"SKEW_RUB": s0}
    mds.Correlations = {}
    mds.Dividends = {}
    mds.Fixings = {}
    mds.RepoSpreadCurves = {}
    disc = mds.RatesCurvesBundles["RUB_RUONIA_OIS"]
    disc.AsOfDate = as_of
    disc.InterpolationMethod = CurveInterpolationMethodEnum.Linear
    disc.RatesSchedule.Dates = [as_of + datetime.timedelta(days=180), as_of + datetime.timedelta(days=365), t_mat]
    disc.RatesSchedule.Rates = [r_zero * 100.0] * 3
    mds.VolatilitySurfaces = {
        "SKEW_RUB": {
            "Identifier": "SKEW_RUB",
            "DayCountConvention": "Act365",
            "ExpirationDates": [(as_of + datetime.timedelta(days=d_exp)).isoformat() for d_exp in exp_days],
            "Strikes": strikes,
            "VolatilityQuotes": quotes,
        }
    }

    st_tmp = preprocess_eq_autocall(req)
    f_rel = float(st_tmp.forward_perf[0, -1].item())
    f_abs = float(st_tmp.dividend_schedule.forward_price[0, -1].item())
    surf = st_tmp.vol_surfaces["SKEW_RUB"]
    tau_l = float(st_tmp.tau_obs[-1].item())
    df_l = float(st_tmp.df_pay[-1].item())

    for mult in (1.0, 1.2):
        r_k = copy.deepcopy(req)
        r_k.Product.Strike = mult * f_rel
        st_k = preprocess_eq_autocall(r_k)
        res = price_eq_autocall_dupire_qmc(
            st_k,
            n_paths=4096,
            n_replicates=8,
            n_substeps_per_period=16,
            precision="fp64_audit",
            cv_mode="bs1d_exact",
            use_gpu=False,
            seed=42,
        )
        iv_mkt = float(surf.iv(tau_l, mult * f_abs))
        w_mkt = iv_mkt * iv_mkt * tau_l
        sw = math.sqrt(w_mkt)
        d1 = (math.log(1.0 / mult) + 0.5 * w_mkt) / sw
        d2 = d1 - sw
        k_rel = mult * f_rel
        b76_put = -df_l * (k_rel * float(ndtr(-d2)) - f_rel * float(ndtr(-d1)))
        vega = df_l * f_rel * math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * math.sqrt(tau_l)
        iv_err_bps = abs(res.product_value - b76_put) / max(vega, 1e-8) * 1e4
        assert iv_err_bps <= 10.0, f"1D skewed LV {mult}F IV error {iv_err_bps:.2f} bps > 10.0 bps"


def test_islongbasket_i_ramp_and_onestar_shifted_spread_rules() -> None:
    """Verify IsLongBasket I_ramp payoff and OneStar ki_barrier/ko_spread rule."""
    from types import SimpleNamespace
    from deepvol.models.autocall.dupire_qmc_solver import _digital_ramp_np, _evaluate_autocall_payoff_np

    # 1. IsLongBasket I_ramp: Payoff_IsLB(T) = Avg_T * ki_wof_K + 1.0 * (1 - ki_wof_K)
    st_lb = SimpleNamespace(
        seasoned_state=SimpleNamespace(c_mem_0=0.0, i_ki_past=0.0),
        callability_index="WorstOf",
        coupon_index="WorstOf",
        option_index="WorstOf",
        is_long_basket=True,
        coupon_barrier_direction="Up",
        coupon_barrier_type="In",
        ki_barrier=1.0,
        ki_spread=-0.10,
        ko_spread=-0.05,
        cpn_spread=0.0,
        force_coupons_on_autocall=False,
        is_memory=False,
        autocall_coupon=0.0,
        no_notional_repayment=False,
        one_star_strike=None,
        one_star_coupon=0.0,
        strike=0.90,
        option_gearing=1.111111,
        lower_strike=None,
        participation_call_strike=None,
        participation_call_rate=0.0,
    )
    # Path 0: Wof=0.85, Avg=0.90 -> ki_wof_K=1 - ramp(0.85, 0.90, -0.10)=0.5 -> Payoff_IsLB = 0.90*0.5 + 1.0*0.5 = 0.95 -> opt_pay=-0.05
    # Path 1: Wof=0.75, Avg=0.80 -> ki_wof_K=1.0 -> Payoff_IsLB = 0.80 -> opt_pay=-0.20
    # Path 2: Wof=0.95, Avg=1.05 -> ki_wof_K=0.0 -> Payoff_IsLB = 1.00 -> opt_pay=0.00
    perf_lb = np.array(
        [
            [[0.85, 0.95]],
            [[0.75, 0.85]],
            [[0.95, 1.15]],
        ],
        dtype=np.float64,
    )
    df_pay = np.array([0.80], dtype=np.float64)
    total_pv, opt_pv, *_ = _evaluate_autocall_payoff_np(
        st_lb,
        perf_lb,
        df_pay=df_pay,
        ko_barriers=np.array([999.0]),
        is_callable=np.array([0.0]),
        cpn_barriers=np.array([999.0]),
        is_guaranteed_cpn=np.array([0.0]),
        coupon_amounts=np.array([0.0]),
        w_fund=np.array([0.0]),
        inv_lex_perm=None,
    )
    expected_opt = df_pay[0] * np.array([-0.05, -0.20, 0.0], dtype=np.float64)
    expected_tot = df_pay[0] * np.array([0.95, 0.80, 1.00], dtype=np.float64)
    np.testing.assert_allclose(opt_pv, expected_opt, atol=1e-12)
    np.testing.assert_allclose(total_pv, expected_tot, atol=1e-12)

    # 2. OneStar shifted left-spread rule when ki_barrier is not None and ko_spread < 0
    st_os_ki = SimpleNamespace(
        seasoned_state=SimpleNamespace(c_mem_0=0.0, i_ki_past=0.0),
        callability_index="WorstOf",
        coupon_index="WorstOf",
        option_index="WorstOf",
        is_long_basket=False,
        coupon_barrier_direction="Up",
        coupon_barrier_type="In",
        ki_barrier=0.85,
        ki_spread=-0.10,
        ko_spread=-0.10,
        cpn_spread=0.0,
        force_coupons_on_autocall=False,
        is_memory=False,
        autocall_coupon=0.0,
        no_notional_repayment=False,
        one_star_strike=1.10,
        one_star_coupon=0.10,
        strike=1.00,
        option_gearing=1.0,
        lower_strike=None,
        participation_call_strike=None,
        participation_call_rate=0.0,
    )
    # BestOf = 1.05 sits at midpoint of [1.00, 1.10] -> star_ind = 0.5
    perf_os = np.array([[[0.70, 1.05]]], dtype=np.float64)
    _, _, _, _, _, _, _, l_os_ki, p_os_ki = _evaluate_autocall_payoff_np(
        st_os_ki,
        perf_os,
        df_pay=np.array([1.0]),
        ko_barriers=np.array([999.0]),
        is_callable=np.array([0.0]),
        cpn_barriers=np.array([999.0]),
        is_guaranteed_cpn=np.array([0.0]),
        coupon_amounts=np.array([0.0]),
        w_fund=np.array([0.0]),
        inv_lex_perm=None,
    )
    assert p_os_ki == pytest.approx(0.5, abs=1e-12)
    assert l_os_ki == pytest.approx(0.05, abs=1e-12)

    # With ki_barrier=None, falls back to narrow centered 0.02 ramp around 1.10 -> star_ind(1.05) = 0.0
    st_os_noki = copy.copy(st_os_ki)
    st_os_noki.ki_barrier = None
    _, _, _, _, _, _, _, l_os_noki, p_os_noki = _evaluate_autocall_payoff_np(
        st_os_noki,
        perf_os,
        df_pay=np.array([1.0]),
        ko_barriers=np.array([999.0]),
        is_callable=np.array([0.0]),
        cpn_barriers=np.array([999.0]),
        is_guaranteed_cpn=np.array([0.0]),
        coupon_amounts=np.array([0.0]),
        w_fund=np.array([0.0]),
        inv_lex_perm=None,
    )
    assert p_os_noki == pytest.approx(0.0, abs=1e-12)
    assert l_os_noki == pytest.approx(0.0, abs=1e-12)
    assert _digital_ramp_np(np.array([1.05]), 1.10, -0.10)[0] == pytest.approx(0.5, abs=1e-12)


def test_ki_obs_type_4_mode_dispatch_shevchenko_and_bgk(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify None->European default and 4-mode ki_obs_type dispatch with Shevchenko/BGK."""
    import datetime

    req_orig, _ = parsed_dag
    req_ki = copy.deepcopy(req_orig)
    as_of_ki = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    req_ki.Product.StartDate = as_of_ki
    req_ki.Product.ObservationDates = [as_of_ki + datetime.timedelta(days=90 * k) for k in range(1, 5)]
    req_ki.Product.PaymentDates = list(req_ki.Product.ObservationDates)
    req_ki.Product.AutoCallTriggerLevels = [1.05] * 4
    req_ki.Product.CouponBarrierLevels = [0.85] * 4
    req_ki.Model.MarketDataSet.AsOfDate = as_of_ki
    req_ki.Model.MarketDataSet.Fixings = {}
    for c in req_ki.Model.MarketDataSet.RatesCurvesBundles.values():
        c.AsOfDate = as_of_ki
    req_ki.Product.OptionDownAndInBarrierLevel = 0.85
    req_ki.Product.OptionDownAndInBarrierObservationType = "European"
    st_eu = preprocess_eq_autocall(req_ki)
    assert float(st_eu.seasoned_state.i_ki_past) < 0.5

    # None ki_obs_type must default to European even if ki_monitored_past=True
    st_none = copy.copy(st_eu)
    st_none.seasoned_state = copy.copy(st_eu.seasoned_state)
    if hasattr(st_none, "ki_obs_type"):
        st_none.ki_obs_type = None
    if hasattr(st_none.seasoned_state, "ki_obs_type"):
        st_none.seasoned_state.ki_obs_type = None
    st_none.seasoned_state.ki_monitored_past = True

    st_disc = copy.copy(st_eu)
    st_disc.seasoned_state = copy.copy(st_eu.seasoned_state)
    st_disc.ki_obs_type = "Discrete"
    st_daily = copy.copy(st_eu)
    st_daily.seasoned_state = copy.copy(st_eu.seasoned_state)
    st_daily.ki_obs_type = "Daily"
    st_cont = copy.copy(st_eu)
    st_cont.seasoned_state = copy.copy(st_eu.seasoned_state)
    st_cont.ki_obs_type = "Continuous"

    r_eu = price_eq_autocall_dupire_qmc(st_eu, n_paths=1024, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=19)
    r_none = price_eq_autocall_dupire_qmc(st_none, n_paths=1024, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=19)
    r_disc = price_eq_autocall_dupire_qmc(st_disc, n_paths=1024, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=19)
    r_daily = price_eq_autocall_dupire_qmc(st_daily, n_paths=1024, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=19)
    r_cont = price_eq_autocall_dupire_qmc(st_cont, n_paths=1024, n_replicates=4, n_substeps_per_period=4, use_gpu=False, seed=19)

    assert r_none.l_put == pytest.approx(r_eu.l_put, abs=1e-12)
    assert r_eu.l_put <= r_disc.l_put + 1e-12
    assert r_disc.l_put <= r_cont.l_put + 1e-12
    assert r_daily.l_put <= r_cont.l_put + 1e-12
    assert r_daily.l_put > r_eu.l_put


def test_repair_arbitrage_quotes_and_canonical_tier3_config(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verify repair_arbitrage_quotes (incl. j==0 guard) and CanonicalTier3Config consumer."""
    from dataclasses import dataclass
    from deepvol.models.autocall.dupire_qmc_solver import repair_arbitrage_quotes

    # 1. Synthetic arbitrage surface with upward vol jump at high strikes + clean slice
    t_exp = np.array([0.25, 1.0], dtype=np.float64)
    strikes = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], dtype=np.float64)
    quotes = np.array(
        [
            [0.32, 0.30, 0.28, 0.27, 0.36, 0.35, 0.34],
            [0.31, 0.29, 0.275, 0.265, 0.35, 0.34, 0.33],
        ],
        dtype=np.float64,
    )
    repaired, info = repair_arbitrage_quotes(strikes, quotes, t_exp, k_ref=4.0)
    assert info["repaired"] is True
    assert info["repaired_slices"] == [0, 1]
    assert repaired.shape == quotes.shape
    surf = RegularizedVolSurface(
        {"Identifier": "SYN_REP", "tenors": t_exp, "Strikes": strikes, "VolatilityQuotes": repaired}
    )
    assert np.all(np.isfinite(surf.quotes))

    # 2. j==0 slope violation must not wrap d2c[-1]
    strikes_j0 = np.array([3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5], dtype=np.float64)
    q_j0 = np.array([[0.10, 0.45, 0.30, 0.28, 0.27, 0.26, 0.25]], dtype=np.float64)
    rep_j0, info_j0 = repair_arbitrage_quotes(strikes_j0, q_j0, np.array([1.0]), k_ref=4.0)
    assert info_j0["j_anom_by_slice"] == [0]
    assert info_j0["repaired"] is True
    assert np.all(np.isfinite(rep_j0))

    # 3. CanonicalTier3Config duck-typed consumption
    @dataclass(frozen=True)
    class _CfgStub:
        n_paths: int = 256
        n_replicates: int = 4
        n_substeps: int = 2
        lv_grid_nt: int = 32
        lv_grid_ny: int = 64
        cv_mode: str = "multifidelity_gbm"
        n_cheap_mf: int = 1024
        seed: int = 42
        adaptive_substeps: bool = False

    req_orig, _ = parsed_dag
    st = preprocess_eq_autocall(req_orig)
    res_cfg = price_eq_autocall_dupire_qmc(st, use_gpu=False, config=_CfgStub())
    assert res_cfg.n_paths == 256
    assert np.isfinite(res_cfg.product_value)

    # 4. Optional IRAO_RUB batch check if production-shadow corpus is available
    batch_dir = REPO_ROOT / "docs" / "eq_autocall_ps_batch_b10316f2"
    if batch_dir.exists():
        import json
        from deepvol.models.autocall.dupire_qmc_solver import _surface_year_fractions_batch

        irao_req_path = batch_dir / "requests" / "trade_052_45040676.json"
        if irao_req_path.exists():
            d_irao = json.loads(irao_req_path.read_text())
            req_irao = EQAutocallableTradeRequest.model_validate(d_irao)
            s_irao = req_irao.Model.MarketDataSet.VolatilitySurfaces["IRAO_RUB"]
            t_irao = _surface_year_fractions_batch(
                req_irao.Model.MarketDataSet.AsOfDate,
                s_irao.ExpirationDates,
                s_irao.DayCountConvention.value,
            )
            rep_irao, info_irao = repair_arbitrage_quotes(
                np.asarray(s_irao.Strikes, dtype=np.float64),
                np.asarray(s_irao.VolatilityQuotes, dtype=np.float64),
                t_irao,
                float(req_irao.Model.MarketDataSet.Spots["IRAO_RUB"]),
            )
            assert info_irao["repaired"] is True
            assert info_irao["repaired_slices"] == list(range(7))
            surf_irao = RegularizedVolSurface(
                {
                    "Identifier": "IRAO_RUB",
                    "tenors": t_irao,
                    "Strikes": s_irao.Strikes,
                    "VolatilityQuotes": rep_irao,
                }
            )
            assert surf_irao.n_exp == 7






