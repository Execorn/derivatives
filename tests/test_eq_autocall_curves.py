"""test_eq_autocall_curves.py — Unit tests for Milestone M0 (Task 0.2) float64 yield/repo curve engine & Gate G1.1."""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from deepvol.api.schemas_eq_autocall import (
    CurveInterpolationMethodEnum,
    DayCountConventionEnum,
    DiscountFactorCurveSpec,
    RatesScheduleSpec,
    parse_murex_autocall_dag,
)
from deepvol.market.eq_curves import (
    ClampedYieldCurve,
    RepoCurve,
    year_fraction,
    year_fractions,
)

REQUEST_TXT_PATH = Path("/home/execorn/programming/derivatives/eq_autocallable_request.txt")


def test_G1_1_rub_ruonia_ois_terminal_discount_factor_zero_ulp() -> None:
    """Acceptance Gate G1.1: RUB_RUONIA_OIS DF(2027-02-09) must match 0.9594909887002673 to <= 1e-14 (0 ULP)."""
    req, expected_res = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    curve_spec = req.Model.MarketDataSet.RatesCurvesBundles["RUB_RUONIA_OIS"]
    curve = ClampedYieldCurve(curve_spec)

    terminal_pay_date = datetime.datetime(2027, 2, 9, 0, 0, tzinfo=datetime.timezone.utc)
    target_df = 0.9594909887002673

    df_scalar = curve.df(terminal_pay_date)
    assert df_scalar.dtype == np.float64
    diff = abs(float(df_scalar) - target_df)
    assert diff <= 1e-15, f"Gate G1.1 failed: |df - target_df| = {diff:.3e} > 1e-15"
    assert float(df_scalar) == target_df  # Exact 0-ULP IEEE-754 float64 equality

    # Also verify vectorized evaluation on all 12 product payment dates
    df_vec = curve.df(req.Product.PaymentDates)
    assert df_vec.shape == (12,)
    assert df_vec.dtype == np.float64
    assert float(df_vec[-1]) == target_df
    assert expected_res["CashFlows"][-1]["Amount"] == target_df


def test_G1_1_analytical_spline_weights_and_bucketed_rho_jacobian() -> None:
    """Verifies W_spline partition of unity, exact reconstruction W @ y == r(tau), and d_df_d_pillar vs FD."""
    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    curve_spec = req.Model.MarketDataSet.RatesCurvesBundles["RUB_RUONIA_OIS"]
    curve = ClampedYieldCurve(curve_spec)

    # Test across left-extrapolated (1d), interior (30d, 91d, 109d, 500d, 2000d), and right-extrapolated (6000d) dates
    as_of = curve_spec.AsOfDate
    eval_dates = [
        as_of + datetime.timedelta(days=d)
        for d in (1, 5, 30, 91, 109, 250, 500, 1500, 4000, 6000)
    ]

    w_mat = curve.spline_weights(eval_dates)
    assert w_mat.shape == (len(eval_dates), 26)
    assert w_mat.dtype == np.float64

    # Partition of unity: rows of W_spline must sum to 1.0
    np.testing.assert_allclose(np.sum(w_mat, axis=1), 1.0, atol=1e-14)

    # Left/right flat extrapolation boundary weights
    np.testing.assert_allclose(w_mat[0], np.eye(26)[0], atol=1e-15)
    np.testing.assert_allclose(w_mat[-1], np.eye(26)[-1], atol=1e-15)

    # Exact linear reconstruction: W_spline @ y_pillars == curve.zero_rate(eval_dates)
    r_from_weights = w_mat @ curve.rates
    r_direct = curve.zero_rate(eval_dates)
    np.testing.assert_allclose(r_from_weights, r_direct, atol=1e-15)

    # Verify analytical d_df_d_pillar against central finite differences on percentage pillar quotes
    df_base, d_df_analytic = curve.df_and_d_df_d_pillar(eval_dates)
    np.testing.assert_allclose(df_base, curve.df(eval_dates), atol=1e-15)
    assert d_df_analytic.shape == (len(eval_dates), 26)

    eps_pct = 1e-5
    base_rates = np.array(curve_spec.RatesSchedule.Rates, dtype=np.float64)
    d_df_fd = np.zeros_like(d_df_analytic)

    for k in range(26):
        r_up = base_rates.copy()
        r_dn = base_rates.copy()
        r_up[k] += eps_pct
        r_dn[k] -= eps_pct
        c_up = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_dates=curve_spec.RatesSchedule.Dates,
            rates=r_up,
            dcc=curve_spec.InterpolationDayCountConvention,
            interpolation_method=curve_spec.InterpolationMethod,
        )
        c_dn = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_dates=curve_spec.RatesSchedule.Dates,
            rates=r_dn,
            dcc=curve_spec.InterpolationDayCountConvention,
            interpolation_method=curve_spec.InterpolationMethod,
        )
        d_df_fd[:, k] = (c_up.df(eval_dates) - c_dn.df(eval_dates)) / (2.0 * eps_pct)

    np.testing.assert_allclose(d_df_analytic, d_df_fd, rtol=1e-7, atol=1e-10)


def test_day_count_conventions_precision() -> None:
    """Verifies Act365, Act360, ActAct, Thirty360/30360, and Business252 in float64."""
    d1 = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    d2 = datetime.datetime(2027, 2, 9, 0, 0, tzinfo=datetime.timezone.utc)  # 109 days

    assert year_fraction(d1, d2, "Act365") == pytest.approx(109.0 / 365.0, abs=1e-16)
    assert year_fraction(d1, d2, "Act360") == pytest.approx(109.0 / 360.0, abs=1e-16)
    # Both 2026 and 2027 are non-leap years, so ActAct == Act365
    assert year_fraction(d1, d2, "ActAct") == pytest.approx(109.0 / 365.0, abs=1e-16)

    # Leap-year ActAct test across 2024 (leap) and 2025 (non-leap)
    leap_start = datetime.date(2024, 7, 1)
    leap_end = datetime.date(2025, 7, 1)
    expected_act_act = (184.0 / 366.0) + (181.0 / 365.0)
    assert year_fraction(leap_start, leap_end, "ActAct") == pytest.approx(expected_act_act, abs=1e-15)

    # Thirty360 / 30360 test
    t360_a = year_fraction(datetime.date(2026, 1, 31), datetime.date(2026, 4, 30), "Thirty360")
    t360_b = year_fraction(datetime.date(2026, 1, 31), datetime.date(2026, 4, 30), "30360")
    assert t360_a == pytest.approx(90.0 / 360.0, abs=1e-16)
    assert t360_a == t360_b

    # Business252 test (Monday 2026-10-19 to Monday 2026-10-26 is 5 business days)
    b252 = year_fraction(datetime.date(2026, 10, 19), datetime.date(2026, 10, 26), "Business252")
    assert b252 == pytest.approx(5.0 / 252.0, abs=1e-16)

    # Vectorized year_fractions check
    arr = year_fractions(d1, [d1, d2], "Act365")
    assert arr.dtype == np.float64
    np.testing.assert_allclose(arr, np.array([0.0, 109.0 / 365.0], dtype=np.float64), atol=1e-16)


@pytest.mark.parametrize(
    "method",
    [
        CurveInterpolationMethodEnum.CubicSpline,
        CurveInterpolationMethodEnum.Linear,
        CurveInterpolationMethodEnum.LogLinear,
        CurveInterpolationMethodEnum.FlatLeft,
        CurveInterpolationMethodEnum.FlatRight,
        CurveInterpolationMethodEnum.MonotoneCubic,
        CurveInterpolationMethodEnum.PCHIP,
        CurveInterpolationMethodEnum.Bessel,
        CurveInterpolationMethodEnum.Harmonic,
        CurveInterpolationMethodEnum.PiecewiseConstantForward,
        CurveInterpolationMethodEnum.NaturalCubicSpline,
        CurveInterpolationMethodEnum.NotAKnotCubicSpline,
        CurveInterpolationMethodEnum.ClampedCubicSpline,
        CurveInterpolationMethodEnum.Akima,
        CurveInterpolationMethodEnum.Kruger,
        CurveInterpolationMethodEnum.NelsonSiegel,
        CurveInterpolationMethodEnum.Svensson,
        CurveInterpolationMethodEnum.QuadraticSpline,
        CurveInterpolationMethodEnum.LogDiscountLinear,
    ],
)
def test_all_19_interpolation_methods_and_jacobians(
    method: CurveInterpolationMethodEnum,
) -> None:
    """Verifies all 19 curve interpolation methods run in float64 and produce exact pillar sensitivities."""
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    pillar_days = [30, 90, 180, 365, 730, 1095, 1825]
    pillar_dates = [as_of + datetime.timedelta(days=d) for d in pillar_days]
    rates_pct = [13.5, 13.8, 14.0, 13.7, 13.9, 14.2, 14.5]

    spec = DiscountFactorCurveSpec(
        Identifier="TEST_CURVE",
        Currency="RUB",
        AsOfDate=as_of,
        InterpolationDayCountConvention=DayCountConventionEnum.Act365,
        InterpolationMethod=method,
        RatesSchedule=RatesScheduleSpec(Dates=pillar_dates, Rates=rates_pct),
    )
    curve = ClampedYieldCurve(spec)

    eval_tenors = np.array([0.05, 0.15, 0.35, 0.75, 1.5, 2.5, 6.0], dtype=np.float64)
    r_vals = curve.zero_rate(eval_tenors)
    df_vals, d_df_mat = curve.df_and_d_df_d_pillar(eval_tenors)

    assert r_vals.dtype == np.float64
    assert df_vals.dtype == np.float64
    assert d_df_mat.dtype == np.float64
    assert np.all(np.isfinite(r_vals))
    assert np.all((df_vals > 0.0) & (df_vals <= 1.0))
    assert d_df_mat.shape == (len(eval_tenors), len(pillar_days))

    # Check sensitivity against finite differences
    eps = 1e-5
    d_df_fd = np.zeros_like(d_df_mat)
    for k in range(len(pillar_days)):
        r_up = list(rates_pct)
        r_dn = list(rates_pct)
        r_up[k] += eps
        r_dn[k] -= eps
        c_up = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_dates=pillar_dates,
            rates=r_up,
            interpolation_method=method,
        )
        c_dn = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_dates=pillar_dates,
            rates=r_dn,
            interpolation_method=method,
        )
        d_df_fd[:, k] = (c_up.df(eval_tenors) - c_dn.df(eval_tenors)) / (2.0 * eps)

    np.testing.assert_allclose(d_df_mat, d_df_fd, rtol=1e-5, atol=1e-9)


def test_repo_curve_and_omitted_posirx_rub_fallback() -> None:
    """Verifies RepoCurve on eq_autocallable_request.txt assets and flat-0.0% fallback for POSIRX_RUB."""
    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    mds = req.Model.MarketDataSet
    as_of = mds.AsOfDate
    obs_date = datetime.datetime(2027, 1, 22, 0, 0, tzinfo=datetime.timezone.utc)  # 91 days

    # Present repo curve (VKCO_RUB: -0.25% Act360)
    vkco_repo = RepoCurve(mds.RepoSpreadCurves["VKCO_RUB"], as_of_date=as_of)
    assert not vkco_repo.is_flat_zero
    q_rate = vkco_repo.repo_rate(obs_date)
    int_q = vkco_repo.integrated_repo(obs_date)
    df_q = vkco_repo.repo_df(obs_date)

    assert q_rate.dtype == np.float64
    assert float(q_rate) == pytest.approx(-0.0025, abs=1e-16)
    assert float(int_q) == pytest.approx(-0.0025 * (91.0 / 360.0), abs=1e-16)
    assert float(df_q) == pytest.approx(np.exp(0.0025 * (91.0 / 360.0)), abs=1e-16)

    # Missing repo curve (POSIRX_RUB omitted in RepoSpreadCurves -> flat 0.0%)
    posirx_spec = mds.RepoSpreadCurves.get("POSIRX_RUB")
    assert posirx_spec is None
    posirx_repo = RepoCurve(posirx_spec, as_of_date=as_of)
    assert posirx_repo.is_flat_zero
    assert float(posirx_repo.repo_rate(obs_date)) == 0.0
    assert float(posirx_repo.integrated_repo(obs_date)) == 0.0
    assert float(posirx_repo.repo_df(obs_date)) == 1.0

    # Vectorized evaluation on all 12 observation dates
    q_vec = vkco_repo.integrated_repo(req.Product.ObservationDates)
    assert q_vec.shape == (12,)
    assert q_vec.dtype == np.float64
