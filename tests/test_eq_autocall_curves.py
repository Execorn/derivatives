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

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUEST_TXT_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


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
    # IEEE-754 exp() is not correctly rounded across glibc/SVML builds, so allow <= 2 ULP.
    assert diff <= 5e-16, f"Terminal DF mismatch: |df - target_df| = {diff:.3e} > 5e-16"

    df_vec = curve.df(req.Product.PaymentDates)
    assert df_vec.shape == (12,)
    assert df_vec.dtype == np.float64
    assert float(df_vec[-1]) == float(df_scalar)
    assert abs(float(df_vec[-1]) - target_df) <= 5e-16
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


def test_t1m004_t1m007_t1m015_t1m019_t1m023_ext014_curves_and_dcc() -> None:
    from deepvol.market.eq_curves import year_fraction_batch_epoch_sec

    # T1M-015 & SCH-013: canonical DCCs & slash aliases & epoch-sec vectorized helper
    d0 = datetime.datetime(2024, 1, 31, 0, 0, tzinfo=datetime.timezone.utc)
    d1 = datetime.datetime(2025, 4, 30, 0, 0, tzinfo=datetime.timezone.utc)
    assert year_fraction(d0, d1, "Act365_25") == pytest.approx((d1 - d0).days / 365.25, abs=1e-15)
    assert year_fraction(d0, d1, "Act366_1") == pytest.approx((d1 - d0).days / 366.0, abs=1e-15)
    assert year_fraction(d0, d1, "Act/365") == year_fraction(d0, d1, "Act365")
    assert year_fraction(d0, d1, "Act/Act") == year_fraction(d0, d1, "ActAct")
    assert year_fraction(d0, d1, "30/360") == year_fraction(d0, d1, "Thirty360")
    assert year_fraction(d0, d1, "30E360") == pytest.approx(450.0 / 360.0, abs=1e-15)
    assert year_fraction(d1, d0, "Act365") == pytest.approx(-year_fraction(d0, d1, "Act365"), abs=1e-15)

    t0_s = int(d0.timestamp())
    t1_s = np.array([int(d0.timestamp()), int(d1.timestamp())], dtype=np.int64)
    for dcc_name in ("Act365", "Act360", "ActAct", "Thirty360"):
        vec = year_fraction_batch_epoch_sec(t0_s, t1_s, dcc_name)
        assert vec.shape == (2,)
        assert vec[0] == pytest.approx(0.0, abs=1e-15)
        assert vec[1] == pytest.approx(year_fraction(d0, d1, dcc_name), abs=1e-12)

    # T1M-004: InterpolationSpaceCoordinates without quote heuristics + tau=0 mask
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    tenors = np.array([0.0, 0.25, 1.0, 2.0, 5.0], dtype=np.float64)
    rates_dec = np.array([0.08, 0.085, 0.09, 0.095, 0.10], dtype=np.float64)
    dfs_pil = np.exp(-rates_dec * tenors)
    log_dfs_pil = -rates_dec * tenors

    for space_coord, mode, quotes in (
        ("Rates", "rates", rates_dec),
        ("Rates", "df", dfs_pil),
        ("Rates", "log_df", log_dfs_pil),
        ("LogarithmOfDiscountFactors", "log_df", log_dfs_pil),
        ("LogarithmOfDiscountFactors", "df", dfs_pil),
        ("LogarithmOfDiscountFactors", "rates", rates_dec),
        ("DiscountFactors", "df", dfs_pil),
        ("DiscountFactors", "log_df", log_dfs_pil),
        ("DiscountFactors", "rates", rates_dec),
    ):
        c_sp = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=tenors,
            rates=quotes,
            space_coords=space_coord,
            quote_expression="Standard",
            input_mode=mode,
            interpolation_method="Linear",
        )
        # tau <= 1e-12 sensitivity at pillar 0 must be masked to 0.0 (no 1/0 NaN!)
        df_q, d_df_q = c_sp.df_and_d_df_d_pillar(np.array([0.0, 0.5, 1.5], dtype=np.float64))
        assert np.all(np.isfinite(df_q)) and np.all(np.isfinite(d_df_q))
        assert df_q[0] == pytest.approx(1.0, abs=1e-15)
        assert np.all(np.isfinite(c_sp.zero_rate(np.array([0.0, 0.5], dtype=np.float64))))
        # Immutability check (CHR-010)
        assert c_sp.tenors.flags.writeable is False
        assert c_sp.rates.flags.writeable is False

    # T1M-007 & T1M-019: Linear/FlatLeft/FlatRight extrapolation + MonotoneConvex + FwdFwdLinear + PiecewiseConstantToTheLeft/Right + Bessel + approx aliases
    eval_t = np.array([0.02, 0.10, 0.40, 1.25, 3.0, 7.0], dtype=np.float64)
    for method in (
        "CubicSpline",
        "NaturalCubicSpline",
        "NotAKnotCubicSpline",
        "QuadraticSpline",
        "Linear",
        "LogLinear",
        "FwdFwdLinear",
        "MonotoneConvex",
        "Bessel",
        "PCHIP",
        "Akima",
        "PiecewiseConstantToTheLeft",
        "PiecewiseConstantToTheRight",
        "LME",
        "ExpSpline",
        "Kruger",
    ):
        for extrap in ("Flat", "Linear", "FlatLeft", "FlatRight"):
            c_m = ClampedYieldCurve(
                as_of_date=as_of,
                pillar_tenors=np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float64),
                rates=np.array([12.0, 13.5, 11.0, 14.0, 13.0], dtype=np.float64),
                interpolation_method=method,
                extrapolation_method=extrap,
            )
            r_eval = c_m.zero_rate(eval_t)
            w_eval = c_m.spline_weights(eval_t)
            assert np.all(np.isfinite(r_eval))
            assert np.all(np.isfinite(w_eval))
            if method in ("CubicSpline", "Linear", "Bessel", "FwdFwdLinear", "NaturalCubicSpline", "NotAKnotCubicSpline"):
                np.testing.assert_allclose(w_eval @ c_m.rates, r_eval, atol=1e-12)

    # Single-pillar curve & empty eval
    c_single = ClampedYieldCurve(as_of_date=as_of, pillar_tenors=[1.0], rates=[10.0])
    assert float(c_single.zero_rate(0.5)) == pytest.approx(0.10)
    assert c_single.spline_weights([0.2, 2.0]).shape == (2, 1)
    assert c_single.spline_weights([]).shape == (0, 1)
    assert c_single.df(np.array([], dtype=np.float64)).shape == (0,)

    # EXT-014: Mapping dict construction for ClampedYieldCurve & RepoCurve + T1M-023 RepoCurve tau <= 0 clamp
    dict_curve = ClampedYieldCurve(
        {
            "Identifier": "DICT_RUB",
            "Currency": "RUB",
            "AsOfDate": as_of.isoformat(),
            "InterpolationDayCountConvention": "Act365",
            "InterpolationMethod": "MonotoneConvex",
            "InterpolationSpaceCoordinates": "DiscountFactors",
            "ExtrapolationMethod": "Linear",
            "RatesSchedule": {
                "Dates": [
                    (as_of + datetime.timedelta(days=90)).isoformat(),
                    (as_of + datetime.timedelta(days=365)).isoformat(),
                    (as_of + datetime.timedelta(days=730)).isoformat(),
                ],
                "DiscountFactors": [0.97, 0.90, 0.81],
            },
        }
    )
    assert float(dict_curve.df(as_of + datetime.timedelta(days=365))) == pytest.approx(0.90, abs=1e-12)

    repo_dict = RepoCurve(
        {
            "Identifier": "DICT_REPO",
            "InterpolationDayCountConvention": "Act365_25",
            "RatesSchedule": {"Periods": ["1W", "3M", "1Y"], "Rates": [0.5, 1.0, 1.5]},
        },
        as_of_date=as_of,
    )
    assert float(repo_dict.repo_df(0.0)) == pytest.approx(1.0, abs=1e-15)
    assert float(repo_dict.repo_df(-0.1)) == pytest.approx(1.0, abs=1e-15)
    assert float(repo_dict.integrated_repo(-0.1)) == pytest.approx(0.0, abs=1e-15)
    assert float(repo_dict.repo_df(np.float64(0.5))) < 1.0

    # Validation error checks
    with pytest.raises(ValueError):
        ClampedYieldCurve(as_of_date=as_of, pillar_tenors=[1.0, 0.5], rates=[10.0, 11.0])
    with pytest.raises(ValueError):
        ClampedYieldCurve(as_of_date=as_of, pillar_tenors=[1.0, 2.0], rates=[10.0, 11.0], interpolation_method="BogusMethod")

    # Additional branch-coverage edge cases for eq_curves.py
    from deepvol.market.eq_curves import dcc_denominator

    assert dcc_denominator("Business252") == pytest.approx(252.0)
    assert dcc_denominator("30E360") == pytest.approx(360.0)
    assert dcc_denominator("ActActISDA") == pytest.approx(365.0)
    with pytest.raises(ValueError):
        dcc_denominator("BogusDCC")

    # 3-pillar NotAKnotCubicSpline & Akima fallback + tau=0 handling in spline/basis
    for m_small in ("NotAKnotCubicSpline", "Akima"):
        c3 = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=[0.5, 1.0, 2.0],
            rates=[10.0, 12.0, 11.0],
            interpolation_method=m_small,
        )
        assert np.all(np.isfinite(c3.zero_rate([0.0, 0.75, 3.0])))
        assert c3.spline_weights([0.0, 0.75, 3.0]).shape == (3, 3)

    # input_mode="log_df" and space_coords="LogarithmOfDiscountFactors" / "Rates" with input_mode="df"
    c_logdf = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=[0.5, 1.0, 2.0],
        rates=[-0.05, -0.10, -0.22],
        space_coords="LogarithmOfDiscountFactors",
        quote_expression="Standard",
        interpolation_method="Linear",
    )
    assert float(c_logdf.df(1.0)) == pytest.approx(np.exp(-0.10), abs=1e-12)

    c_df_in = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=[0.5, 1.0, 2.0],
        rates=[0.95, 0.90, 0.80],
        space_coords="DiscountFactors",
        quote_expression="Standard",
        interpolation_method="MonotoneConvex",
    )
    assert np.all(np.isfinite(c_df_in.df([0.0, 0.25, 1.0, 3.0])))
    assert np.all(np.isfinite(c_df_in.spline_weights([0.0, 0.25, 1.5])))

    c_fwd_tau0 = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=[0.5, 1.0, 2.0],
        rates=[10.0, 12.0, 11.0],
        interpolation_method="FwdFwdLinear",
    )
    assert np.all(np.isfinite(c_fwd_tau0.spline_weights([0.0, 0.25, 0.75, 3.0])))
    c_log_tau0 = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=[0.5, 1.0, 2.0],
        rates=[10.0, 12.0, 11.0],
        interpolation_method="LogLinear",
    )
    assert np.all(np.isfinite(c_log_tau0.spline_weights([0.0, 0.25, 0.75, 3.0])))

    repo_dates = RepoCurve(
        {
            "Identifier": "REPO_DATES",
            "RatesSchedule": {
                "Dates": [(as_of + datetime.timedelta(days=90)).isoformat(), (as_of + datetime.timedelta(days=365)).isoformat()],
                "Rates": [0.5, 1.0],
            },
        },
        as_of_date=as_of,
    )
    assert float(repo_dates.repo_rate(as_of + datetime.timedelta(days=90))) == pytest.approx(0.005)
    repo_num = RepoCurve(as_of_date=as_of, periods_days=[90.0, 365.0], rates=[0.5, 1.0])
    assert repo_num.is_flat_zero is False
    assert repo_num.repo_df([]).shape == (0,)
    repo_empty = RepoCurve(spec=None, as_of_date=as_of)
    assert repo_empty.is_flat_zero is True
    assert float(repo_empty.repo_rate(0.5)) == 0.0
    assert float(repo_empty.integrated_repo(0.5)) == 0.0

    with pytest.raises(ValueError):
        ClampedYieldCurve(as_of_date=as_of, pillar_tenors=[1.0, 2.0], rates=[10.0], input_mode="bad_mode")


def test_curves_audit_blind_spots() -> None:
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    rates_pct = np.array([12.0, 12.5, 13.0, 13.4, 13.8], dtype=np.float64)
    rates_dec = rates_pct / 100.0
    log_df_vals = -rates_dec * tenors
    df_vals = np.exp(log_df_vals)
    eval_int = np.array([0.25, 0.35, 0.75, 1.5, 3.5, 5.0], dtype=np.float64)
    eval_t = np.array([0.1, 0.35, 0.75, 1.5, 3.5, 6.0], dtype=np.float64)

    for method in (
        "LogLinear",
        "LogDiscountLinear",
        "PiecewiseConstantForward",
        "FwdFwdLinear",
    ):
        c_rates = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=tenors,
            rates=rates_pct,
            interpolation_method=method,
            space_coords="Rates",
        )
        ref_df_int, ref_ddp_int = c_rates.df_and_d_df_d_pillar(eval_int)

        c_logdf = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=tenors,
            rates=log_df_vals,
            interpolation_method=method,
            space_coords="LogarithmOfDiscountFactors",
            quote_expression="Standard",
        )
        c_df = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=tenors,
            rates=df_vals,
            interpolation_method=method,
            space_coords="DiscountFactors",
            quote_expression="Standard",
        )

        # Verify interior DF parity AND Jacobian sign against exact equivalent Rates parameterization.
        np.testing.assert_allclose(c_logdf.df(eval_int), ref_df_int, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(c_df.df(eval_int), ref_df_int, rtol=1e-12, atol=1e-12)
        assert np.all(np.sum(ref_ddp_int, axis=1) <= -1e-12)
        if method != "FwdFwdLinear":
            assert np.all(ref_ddp_int <= 1e-14)

        for c_alt, quotes_alt in ((c_logdf, log_df_vals), (c_df, df_vals)):
            vals_alt, jac_alt = c_alt.df_and_d_df_d_pillar(eval_t)
            np.testing.assert_allclose(c_alt.df(eval_int), ref_df_int, rtol=1e-12, atol=1e-12)
            eps = 1e-6
            fd_jac = np.zeros_like(jac_alt)
            for k in range(len(quotes_alt)):
                q_up = quotes_alt.copy()
                q_dn = quotes_alt.copy()
                q_up[k] += eps
                q_dn[k] -= eps
                df_u = ClampedYieldCurve(
                    as_of_date=as_of,
                    pillar_tenors=tenors,
                    rates=q_up,
                    interpolation_method=method,
                    space_coords=c_alt.space_coords,
                    quote_expression="Standard",
                ).df(eval_t)
                df_d = ClampedYieldCurve(
                    as_of_date=as_of,
                    pillar_tenors=tenors,
                    rates=q_dn,
                    interpolation_method=method,
                    space_coords=c_alt.space_coords,
                    quote_expression="Standard",
                ).df(eval_t)
                fd_jac[:, k] = (df_u - df_d) / (2.0 * eps)
            np.testing.assert_allclose(jac_alt, fd_jac, rtol=1e-6, atol=1e-9)

    # CurveInterpolationMethodEnum.Smooth support and FD Jacobian consistency.
    c_smooth = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=tenors,
        rates=rates_pct,
        interpolation_method=CurveInterpolationMethodEnum.Smooth,
    )
    sm_df, sm_ddp = c_smooth.df_and_d_df_d_pillar(eval_t)
    assert np.all(np.isfinite(sm_df)) and np.all((sm_df > 0.0) & (sm_df <= 1.0))
    np.testing.assert_allclose(c_smooth.zero_rate(tenors), rates_dec, atol=1e-12)
    np.testing.assert_allclose(np.sum(c_smooth.spline_weights(eval_t), axis=1), 1.0, atol=1e-12)
    fd_sm = np.zeros_like(sm_ddp)
    for k in range(len(rates_pct)):
        ru, rd = rates_pct.copy(), rates_pct.copy()
        ru[k] += 1e-5
        rd[k] -= 1e-5
        df_u = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=tenors,
            rates=ru,
            interpolation_method=CurveInterpolationMethodEnum.Smooth,
        ).df(eval_t)
        df_d = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=tenors,
            rates=rd,
            interpolation_method=CurveInterpolationMethodEnum.Smooth,
        ).df(eval_t)
        fd_sm[:, k] = (df_u - df_d) / (2.0 * 1e-5)
    np.testing.assert_allclose(sm_ddp, fd_sm, rtol=1e-6, atol=1e-9)

    # MonotoneConvex C^0 instantaneous forward continuity on stress knot geometries.
    for stress_r in ([1.0, 2.0, 3.0, 6.0], [5.0, 5.0, 6.0, 30.0], [18.0, 4.0, 12.0, 2.0]):
        stress_t = np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float64)
        c_mc = ClampedYieldCurve(
            as_of_date=as_of,
            pillar_tenors=stress_t,
            rates=stress_r,
            interpolation_method="MonotoneConvex",
        )
        dt_probe = 1e-6
        for t_knot in stress_t[1:-1]:
            df_m = float(c_mc.df(t_knot - dt_probe))
            df_0 = float(c_mc.df(t_knot))
            df_p = float(c_mc.df(t_knot + dt_probe))
            f_left = -(np.log(df_0) - np.log(df_m)) / dt_probe
            f_right = -(np.log(df_p) - np.log(df_0)) / dt_probe
            assert abs(f_right - f_left) < 1e-3
        np.testing.assert_allclose(
            c_mc.df(stress_t), np.exp(-(np.asarray(stress_r, dtype=np.float64) / 100.0) * stress_t), atol=1e-12
        )
        grid_t = np.linspace(0.01, 2.0, 200)
        df_grid = c_mc.df(grid_t)
        assert np.all(np.isfinite(df_grid)) and np.all(df_grid > 0.0)
        if np.all(np.diff((np.asarray(stress_r, dtype=np.float64) / 100.0) * stress_t) > 0.0):
            assert np.all(np.diff(np.log(df_grid)) <= 1e-12)

    # Rates=[] + Spreads fallback, RepoCurve edge inputs, NaN validation, and dotted DCC normalization.
    c_spread_only = ClampedYieldCurve(
        DiscountFactorCurveSpec(
            Identifier="SPREAD_ONLY",
            Currency="RUB",
            AsOfDate=as_of,
            InterpolationDayCountConvention=DayCountConventionEnum.Act365,
            InterpolationMethod=CurveInterpolationMethodEnum.Linear,
            RatesSchedule=RatesScheduleSpec(
                Dates=[
                    as_of + datetime.timedelta(days=30),
                    as_of + datetime.timedelta(days=365),
                ],
                Rates=[],
                Spreads=[12.0, 13.0],
            ),
        )
    )
    assert np.all(np.isfinite(c_spread_only.df([0.25, 1.0])))

    with pytest.raises(ValueError, match="repo period"):
        RepoCurve(
            {
                "Identifier": "REPO_NEG",
                "AsOfDate": as_of,
                "RatesSchedule": {
                    "Periods": ["-30D", "30D", "1Y"],
                    "Rates": [0.4, 0.6, 0.8],
                },
            }
        )
    repo_edge = RepoCurve(
        {
            "Identifier": "REPO_EDGE",
            "AsOfDate": as_of,
            "RatesSchedule": {
                "Periods": ["30D", "1Y"],
                "Rates": [0.6, 0.8],
            },
        }
    )
    assert np.isfinite(float(repo_edge.repo_rate(np.int64(90) / 365.0)))
    assert float(RepoCurve({"Identifier": "REPO_EMPTY", "RatesSchedule": {"Periods": [], "Rates": []}}, as_of_date=as_of).repo_rate(0.5)) == 0.0

    for bad_t, bad_r in (([0.5, float("nan")], [10.0, 11.0]), ([0.5, 1.0], [10.0, float("inf")])):
        with pytest.raises(ValueError):
            ClampedYieldCurve(as_of_date=as_of, pillar_tenors=bad_t, rates=bad_r)

    assert year_fraction(as_of, as_of + datetime.timedelta(days=365), "Act/365.25") == pytest.approx(
        365.0 / 365.25, rel=1e-12
    )
    assert year_fraction(as_of, as_of + datetime.timedelta(days=365), "Act.365") == pytest.approx(1.0, rel=1e-12)


def test_monotone_cubic_rates_linear_parity_and_zero_spillover() -> None:
    """43350563 parity: MonotoneCubic in Rates space uses piecewise-linear zero-rate interpolation."""
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    tenors = np.array([0.0027397260273972603, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    rates_pct = np.array([15.5, 14.9, 14.4, 14.1, 14.15, 14.3, 14.6], dtype=np.float64)

    c_mc = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=tenors,
        rates=rates_pct,
        interpolation_method="MonotoneCubic",
        space_coords="Rates",
    )
    c_lin = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=tenors,
        rates=rates_pct,
        interpolation_method="Linear",
        space_coords="Rates",
    )

    eval_t = np.array([0.001, 0.10, 0.37, 0.80, 1.50, 2.25, 4.0, 6.5], dtype=np.float64)
    np.testing.assert_allclose(c_mc.zero_rate(eval_t), c_lin.zero_rate(eval_t), atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(c_mc.df(eval_t), c_lin.df(eval_t), atol=1e-15, rtol=0.0)

    # Zero Rho spillover onto non-adjacent pillars for an interior evaluation point in (2.0, 3.0)
    w_mid = c_mc.spline_weights(np.array([2.25], dtype=np.float64))[0]
    np.testing.assert_allclose(w_mid[:4], 0.0, atol=1e-15)
    np.testing.assert_allclose(w_mid[6:], 0.0, atol=1e-15)
    assert w_mid[4] == pytest.approx(0.75, abs=1e-15)
    assert w_mid[5] == pytest.approx(0.25, abs=1e-15)

    # DiscountFactors space preserves non-linear PCHIP behavior cleanly
    dfs_pil = np.exp(-(rates_pct / 100.0) * tenors)
    c_mc_df = ClampedYieldCurve(
        as_of_date=as_of,
        pillar_tenors=tenors,
        rates=dfs_pil,
        interpolation_method="MonotoneCubic",
        space_coords="DiscountFactors",
        quote_expression="Standard",
    )
    df_vals, d_df_mat = c_mc_df.df_and_d_df_d_pillar(eval_t)
    assert np.all(np.isfinite(df_vals)) and np.all(np.isfinite(d_df_mat))




