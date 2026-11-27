"""test_eq_autocall_schemas.py — Unit tests for Milestone M0 (Task 0.1) Pydantic V2 schemas & Murex DAG parser."""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from deepvol.api.schemas_eq_autocall import (
    AutocallQueryMetric,
    AutoCallableProductSpec,
    CashFlowRow,
    CorrelationPairSpec,
    CurveInterpolationMethodEnum,
    DayCountConventionEnum,
    DiscountFactorCurveSpec,
    DividendsSpec,
    DupireModelSpec,
    EQAutocallableTradeRequest,
    EQAutocallableTradeResponse,
    EquityVolSmileCollectionSpec,
    FixingsSpec,
    MarketDataSetSpec,
    MonitoringIndexEnum,
    PredefinedDividendsSpec,
    RatesScheduleSpec,
    RepoScheduleSpec,
    RepoSpreadCurveSpec,
    parse_murex_autocall_dag,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUEST_TXT_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else REPO_ROOT / "eq_autocallable_request.txt"
)


def test_enums_completeness() -> None:
    """Verifies all required enum values across Queries, MonitoringIndex, DCC, and 19 CurveInterpolationMethods."""
    expected_queries = {
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
    }
    actual_queries = {m.value for m in AutocallQueryMetric}
    assert expected_queries.issubset(actual_queries)

    expected_indices = {
        "WorstOf",
        "BestOf",
        "Average",
        "Outperformance",
        "TwinWin",
        "Rainbow",
        "Lookback",
        "Spread",
    }
    actual_indices = {m.value for m in MonitoringIndexEnum}
    assert expected_indices.issubset(actual_indices)

    expected_dccs = {"Act365", "Act360", "ActAct", "Thirty360", "30360", "Business252"}
    actual_dccs = {m.value for m in DayCountConventionEnum}
    assert expected_dccs.issubset(actual_dccs)

    expected_19_interps = {
        "CubicSpline",
        "Linear",
        "LogLinear",
        "FlatLeft",
        "FlatRight",
        "MonotoneCubic",
        "PCHIP",
        "Bessel",
        "Harmonic",
        "PiecewiseConstantForward",
        "NaturalCubicSpline",
        "NotAKnotCubicSpline",
        "ClampedCubicSpline",
        "Akima",
        "Kruger",
        "NelsonSiegel",
        "Svensson",
        "QuadraticSpline",
        "LogDiscountLinear",
    }
    actual_interps = {m.value for m in CurveInterpolationMethodEnum}
    assert expected_19_interps.issubset(actual_interps)


def test_parse_murex_autocall_dag_production_fixture() -> None:
    """Verifies 0-error parsing of production autocall request fixture."""
    raw_text = REQUEST_TXT_PATH.read_text(encoding="utf-8")
    req, expected_res = parse_murex_autocall_dag(raw_text)

    # Verify root request metadata and queries
    assert req.TradeInfo == "45476647/EMTN_CIB_SEC"
    assert req.Queries == [AutocallQueryMetric.ProductValue, AutocallQueryMetric.CashFlows]

    # Verify Tier 0 stripped fields audit log
    for stripped_name in (
        "CalculationMethod",
        "NumberOfPaths",
        "StaticDataSet",
        "GreekMetricsSettings",
        "DifferentiationMethod",
        "BatchSizeForMCPath",
        "UseNumeraireCurrencyParameterForCrossCurrency",
    ):
        assert stripped_name in req.stripped_legacy_fields

    # Verify Product terms
    prod = req.Product
    assert prod.PaymentCurrency == "RUB"
    assert prod.DayCountConvention == DayCountConventionEnum.Act360
    assert prod.Strike == pytest.approx(0.9)
    assert prod.OptionGearing == pytest.approx(1.11111)
    assert prod.FixedCouponAmount == pytest.approx(0.04688)
    assert prod.FixedRate is None
    assert prod.IsMemory is True
    assert prod.CallabilityMonitoringIndex == MonitoringIndexEnum.WorstOf
    assert len(prod.ObservationDates) == 12
    assert len(prod.PaymentDates) == 12
    assert len(prod.AutoCallTriggerLevels) == 12
    assert prod.CouponBarrierLevels is not None and len(prod.CouponBarrierLevels) == 12
    assert prod.get_coupon_rates() == pytest.approx([0.04688] * 12)
    assert prod.UnderlyingIdentifiers == [
        "MAGN_RUB",
        "POSIRX_RUB",
        "SIBN_RUB",
        "TATN_RUB",
        "VKCO_RUB",
    ]
    assert prod.UnderlyingReferenceSpots == pytest.approx([58.4, 2220.0, 841.95, 694.9, 661.0])

    # Verify MarketDataSet and nested model types
    assert isinstance(req.Model, DupireModelSpec)
    mds = req.Model.MarketDataSet
    assert isinstance(mds, MarketDataSetSpec)
    assert mds.AsOfDate == datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    assert "RUB_RUONIA_OIS" in mds.RatesCurvesBundles
    rub_curve = mds.RatesCurvesBundles["RUB_RUONIA_OIS"]
    assert isinstance(rub_curve, DiscountFactorCurveSpec)
    assert isinstance(rub_curve.RatesSchedule, RatesScheduleSpec)
    assert len(rub_curve.RatesSchedule.Dates) == 26
    assert set(mds.RepoSpreadCurves.keys()) == {"VKCO_RUB", "SIBN_RUB", "MAGN_RUB", "TATN_RUB"}
    assert isinstance(mds.RepoSpreadCurves["VKCO_RUB"], RepoSpreadCurveSpec)
    assert isinstance(mds.RepoSpreadCurves["VKCO_RUB"].RatesSchedule, RepoScheduleSpec)
    assert "POSIRX_RUB" not in mds.RepoSpreadCurves
    assert len(mds.VolatilitySurfaces) == 5
    assert isinstance(mds.VolatilitySurfaces["VKCO_RUB"], EquityVolSmileCollectionSpec)
    assert len(mds.Dividends) == 5
    assert isinstance(mds.Dividends["VKCO_RUB"], DividendsSpec)
    assert len(mds.Fixings) == 5
    assert isinstance(mds.Fixings["VKCO_RUB"], FixingsSpec)

    # Verify Correlation Matrix extraction
    corr_mat = mds.get_correlation_matrix(prod.UnderlyingIdentifiers)
    assert corr_mat.shape == (5, 5)
    assert corr_mat.dtype == np.float64
    assert np.allclose(corr_mat, corr_mat.T)
    assert np.allclose(np.diag(corr_mat), 1.0)
    assert corr_mat[0, 1] == pytest.approx(0.59)  # MAGN_RUB & POSIRX_RUB
    assert corr_mat[2, 3] == pytest.approx(0.70)  # SIBN_RUB & TATN_RUB
    eigvals = np.linalg.eigvalsh(corr_mat)
    assert np.all(eigvals > 0.0)

    # Verify expected result dictionary extracted from fixture
    assert expected_res["ProductValue"] == pytest.approx(0.2181930543553945, abs=1e-15)
    assert expected_res["Option"] == pytest.approx(-0.7412979343448728, abs=1e-15)
    assert expected_res["StandardError"] == pytest.approx(0.0002913772720117921, abs=1e-15)
    assert expected_res["ReportingCurrency"] == "RUB"
    assert expected_res["TradeInfo"] == "45476647/EMTN_CIB_SEC"
    assert len(expected_res["CashFlows"]) == 37

    # Validate CashFlows via CashFlowRow and EQAutocallableTradeResponse
    resp = EQAutocallableTradeResponse.model_validate(
        {
            "ProductValue": expected_res["ProductValue"],
            "ReportingCurrency": expected_res["ReportingCurrency"],
            "StandardError": expected_res["StandardError"],
            "CashToSettleToday": expected_res["CashToSettleToday"],
            "CashFlows": expected_res["CashFlows"],
            "TradeInfo": expected_res["TradeInfo"],
        }
    )
    assert len(resp.CashFlows) == 37
    assert isinstance(resp.CashFlows[0], CashFlowRow)


def test_parse_murex_dag_path_and_dict_inputs() -> None:
    """Verifies `parse_murex_autocall_dag` accepts Path, file path string, and already-resolved dicts."""
    req_from_path, res_from_path = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    req_from_str_path, _ = parse_murex_autocall_dag(str(REQUEST_TXT_PATH), allow_file_path=True)
    assert req_from_path.TradeInfo == req_from_str_path.TradeInfo == "45476647/EMTN_CIB_SEC"

    dumped = req_from_path.model_dump()
    dumped["Result"] = res_from_path
    req_from_dict, res_from_dict = parse_murex_autocall_dag(dumped)
    assert req_from_dict.Product.Strike == pytest.approx(0.9)
    assert res_from_dict["ProductValue"] == pytest.approx(0.2181930543553945, abs=1e-15)


def test_tier0_legacy_stripping_and_extra_forbid() -> None:
    """Verifies Tier 0 legacy fields are stripped at all levels while unknown typos raise ValidationError."""
    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    base_dict: dict[str, Any] = req.model_dump()
    base_dict.pop("stripped_legacy_fields", None)

    # Inject all 17 Tier 0 legacy fields across root, GreekMetricsSettings, Model, and Curve
    base_dict["CalculationMethod"] = "MonteCarloWithAAD"
    base_dict["NumberOfPaths"] = 65536
    base_dict["BatchSizeForMCPath"] = 16384
    base_dict["StaticDataSet"] = "deadbeef" * 5
    base_dict["GreekMetricsSettings"] = {
        "DifferentiationMethod": "Bumps",
        "BatchSizeForMCPath": 8192,
        "NumberGenerationMethod": "Sobol",
        "NumberGenerationClass": "craft",
        "PathGenerationMethod": "BrownianBridge",
        "BumpParametersOverrides": {
            "Delta": {
                "MetricMapping": "SpotUpDown",
                "StepSize": [-0.01, 0.01],
            }
        },
    }
    base_dict["Model"]["UseNumeraireCurrencyParameterForCrossCurrency"] = False
    curve_dict = base_dict["Model"]["MarketDataSet"]["RatesCurvesBundles"]["RUB_RUONIA_OIS"]
    curve_dict["InterpolationLeftLength"] = 0
    curve_dict["SpreadType"] = "RatesAdditive"
    curve_dict["BaseCurve"] = "01234567" * 5

    validated = EQAutocallableTradeRequest.model_validate(base_dict)
    expected_stripped = {
        "CalculationMethod",
        "NumberOfPaths",
        "BatchSizeForMCPath",
        "StaticDataSet",
        "GreekMetricsSettings",
        "DifferentiationMethod",
        "BumpParametersOverrides",
        "MetricMapping",
        "StepSize",
        "NumberGenerationMethod",
        "NumberGenerationClass",
        "PathGenerationMethod",
        "UseNumeraireCurrencyParameterForCrossCurrency",
        "InterpolationLeftLength",
        "SpreadType",
        "BaseCurve",
    }
    assert expected_stripped.issubset(set(validated.stripped_legacy_fields))

    # Unknown typo at root must fail
    bad_root = copy.deepcopy(base_dict)
    bad_root["InvalidRandomField"] = 42
    with pytest.raises(ValidationError):
        EQAutocallableTradeRequest.model_validate(bad_root)

    # Unknown typo inside GreekMetricsSettings must also fail
    bad_gk = copy.deepcopy(base_dict)
    bad_gk["GreekMetricsSettings"]["UnknownGreekOption"] = True
    with pytest.raises(ValidationError):
        EQAutocallableTradeRequest.model_validate(bad_gk)

    # Unknown typo inside Product must fail
    bad_prod = copy.deepcopy(base_dict)
    bad_prod["Product"]["TypoBarrierField"] = 1.0
    with pytest.raises(ValidationError):
        EQAutocallableTradeRequest.model_validate(bad_prod)


def test_fixed_rate_and_fixed_coupon_amount_mutual_exclusivity() -> None:
    """Verifies FixedRate and FixedCouponAmount cannot both be non-None on AutoCallableProductSpec."""
    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    prod_dict = req.Product.model_dump()

    # Setting both FixedRate and FixedCouponAmount must raise ValidationError
    both_dict = copy.deepcopy(prod_dict)
    both_dict["FixedRate"] = 0.15
    both_dict["FixedCouponAmount"] = 0.04688
    with pytest.raises(ValidationError, match="mutually exclusive"):
        AutoCallableProductSpec.model_validate(both_dict)

    # Using FixedRate alone works and broadcasts in .get_coupon_rates()
    rate_only_dict = copy.deepcopy(prod_dict)
    rate_only_dict["FixedCouponAmount"] = None
    rate_only_dict["FixedRate"] = 0.12
    prod_rate = AutoCallableProductSpec.model_validate(rate_only_dict)
    assert prod_rate.get_coupon_rates() == pytest.approx([0.12] * 12)

    # Using per-period list FixedRate works
    vec_dict = copy.deepcopy(prod_dict)
    vec_dict["FixedCouponAmount"] = None
    vec_dict["FixedRate"] = [0.01 * (i + 1) for i in range(12)]
    prod_vec = AutoCallableProductSpec.model_validate(vec_dict)
    assert prod_vec.get_coupon_rates() == pytest.approx([0.01 * (i + 1) for i in range(12)])


def test_dag_ampersand_bang_references_and_bottom_result() -> None:
    """Verifies `parse_murex_autocall_dag` supports `'&<sha1>'`, `'!<sha1>'`, and bottom `Result:` blocks."""
    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    prod_d = req.Product.model_dump()
    mds_d = req.Model.MarketDataSet.model_dump()
    h_root = "a" * 40
    h_prod = "b" * 40
    h_model = "c" * 40
    h_mds = "d" * 40

    dag_dict = {
        h_root: {
            "Product": f"&{h_prod}",
            "Model": f"!{h_model}",
            "CalculationMethod": "MonteCarloWithAAD",
            "NumberOfPaths": 1024,
            "Queries": ["ProductValue", "Delta"],
            "TradeInfo": "SYNTH_001",
        },
        h_prod: prod_d,
        h_model: {
            "TypeName": "DupireLocalVolatilityModel",
            "MarketDataSet": f"&{h_mds}",
            "FinancingCurveIdentifiers": ["RUB_RUONIA_OIS"],
            "UseNumeraireCurrencyParameterForCrossCurrency": False,
        },
        h_mds: mds_d,
        "Result": {
            "ProductValue": 0.55,
            "StandardError": 0.001,
        },
    }
    parsed_req, parsed_res = parse_murex_autocall_dag(dag_dict)
    assert parsed_req.TradeInfo == "SYNTH_001"
    assert parsed_req.Queries == [AutocallQueryMetric.ProductValue, AutocallQueryMetric.Delta]
    assert parsed_res["ProductValue"] == pytest.approx(0.55)
    assert parsed_res["TradeInfo"] == "SYNTH_001"


def test_predefined_dividends_and_correlation_pair_spec() -> None:
    """Verifies PredefinedDividendsSpec and CorrelationPairSpec validation."""
    pdiv = PredefinedDividendsSpec(
        ExDates=[datetime.date(2027, 1, 15), datetime.date(2027, 6, 15)],
        Amounts=[5.0, 7.5],
    )
    assert len(pdiv.ExDates) == 2
    assert pdiv.ExDates[0].tzinfo == datetime.timezone.utc

    cp = CorrelationPairSpec.from_key("VKCO_RUB&MAGN_RUB", 0.55)
    assert cp.key == "MAGN_RUB&VKCO_RUB"
    assert cp.correlation == pytest.approx(0.55)

    with pytest.raises(ValueError, match="Invalid correlation key"):
        CorrelationPairSpec.from_key("INVALID_KEY_WITHOUT_AMPERSAND", 0.5)


def test_sch011_sch018_sch019_ext003_schema_and_dag_edge_cases(tmp_path: Path) -> None:
    from deepvol.api.schemas_eq_autocall import (
        FXVolatilitySmileCollectionSpec,
        _DAGTokenStream,
        _parse_dag_value,
        _parse_datetime_literal,
        _parse_scalar_token,
        _tokenize_dag_fast,
    )

    # SCH-018: nested-paren tzinfo, microseconds, date literal, Z suffix, naive datetime
    dt_nested = _parse_datetime_literal(
        "datetime.datetime(2026, 10, 23, 12, 30, 45, 123456, tzinfo=datetime.timezone(datetime.timedelta(hours=3)))"
    )
    assert dt_nested == datetime.datetime(2026, 10, 23, 9, 30, 45, 123456, tzinfo=datetime.timezone.utc)
    dt_kw = _parse_datetime_literal(
        "datetime.datetime(year=2026, month=10, day=23, hour=5, minute=0, second=0, tzinfo=timezone(timedelta(seconds=-3600)))"
    )
    assert dt_kw == datetime.datetime(2026, 10, 23, 6, 0, tzinfo=datetime.timezone.utc)
    dt_date = _parse_datetime_literal("datetime.date(2026, 10, 23)")
    assert dt_date == datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    dt_utc_fast = _parse_datetime_literal("datetime.datetime(2026, 10, 23, 1, 2, 3, tzinfo=datetime.timezone.utc)")
    assert dt_utc_fast == datetime.datetime(2026, 10, 23, 1, 2, 3, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValueError, match="Malformed datetime literal"):
        _parse_datetime_literal("not_a_date")
    with pytest.raises(ValueError, match="Unsupported tzinfo literal"):
        _parse_datetime_literal("datetime.datetime(2026, 1, 1, tzinfo=FixedOffset(180))")

    # SCH-013: FX Smooth & extended DCCs
    fx_spec = FXVolatilitySmileCollectionSpec(
        Identifier="USD/RUB",
        AsOfDate=datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc),
        SmileInterpolationMethod="Smooth",
        Deltas=["-25D", "ATM", "25D"],
        Tenors=["O/N", "1Y"],
        VolatilityQuotes=[[0.15, 0.14, 0.16], [0.18, 0.17, 0.19]],
    )
    assert fx_spec.SmileInterpolationMethod == CurveInterpolationMethodEnum.Smooth
    for dcc_tok in ("Act365_25", "Act366_1", "Act366_2", "Act366_3", "30E360"):
        assert dcc_tok in {m.value for m in DayCountConventionEnum}

    # Token stream & scalar token branches
    sample_dag_text = (
        "{ 'a': 123, 'b': -1.5e-2, 'c': <skipped>, 'd': True, 'e': false, "
        "'f': None, 'g': datetime.date(2026, 5, 1), 'h': ['x', 0123456789abcdef0123456789abcdef01234567] }"
    )
    stream = _DAGTokenStream(sample_dag_text)
    assert stream.peek() is not None
    parsed_stream = _parse_dag_value(stream)
    assert parsed_stream["a"] == 123
    assert parsed_stream["b"] == pytest.approx(-0.015)
    assert parsed_stream["c"] is None
    assert parsed_stream["d"] is True
    assert parsed_stream["e"] is False
    assert parsed_stream["f"] is None
    assert parsed_stream["g"].year == 2026
    assert len(_tokenize_dag_fast("{ 'k': 1 }")) == 5
    assert _parse_scalar_token("0123456789abcdef0123456789abcdef01234567") == "0123456789abcdef0123456789abcdef01234567"
    assert _parse_scalar_token("9" * 40) == "9" * 40
    assert _parse_scalar_token("123456") == 123456
    assert _parse_scalar_token("true") is True and _parse_scalar_token("null") is None
    assert _parse_scalar_token("@" + "a" * 40) == "@" + "a" * 40
    with pytest.raises(ValueError):
        _parse_scalar_token("'unclosed")
    with pytest.raises(ValueError):
        _parse_scalar_token("}")
    with pytest.raises(ValueError, match="COLON"):
        _parse_dag_value(_DAGTokenStream("{ 'k' 1 }"))

    # SCH-011: Circular DAG & dangling Tier-0 vs missing non-Tier-0 reference
    req, res_ref = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    base_d = req.model_dump()
    base_d.pop("stripped_legacy_fields", None)
    h_root = "1" * 40
    h_cyc = "2" * 40
    with pytest.raises(ValueError, match="Circular dependency"):
        parse_murex_autocall_dag(
            {
                h_root: {"Product": f"&{h_cyc}", "Model": base_d["Model"]},
                h_cyc: {"Product": f"&{h_root}"},
            }
        )
    with pytest.raises(KeyError, match="Dangling Murex DAG reference"):
        parse_murex_autocall_dag(
            {
                h_root: {"Product": "&" + "f" * 40, "Model": base_d["Model"]},
            }
        )

    # Dangling Tier-0 / external ID keys should pass without KeyError
    tier0_d = req.model_dump(mode="json")
    tier0_d.pop("stripped_legacy_fields", None)
    tier0_d["StaticDataSet"] = "@" + "e" * 40
    tier0_d["Model"]["MarketDataSet"]["RatesCurvesBundles"]["RUB_RUONIA_OIS"]["BaseCurve"] = "@" + "d" * 40
    dag_file = tmp_path / "dag_sample.txt"
    h_prod = "3" * 40
    h_mds = "4" * 40
    h_mod = "5" * 40
    dag_file.write_text(
        f"@{h_root}\n"
        f"{h_root} class: ConstantMessageMap\n"
        f"{{ Product: &{h_prod}, Model: !{h_mod}, Queries: ['ProductValue'], StaticDataSet: '@{'e' * 40}' }}\n"
        f"{h_prod} : {repr(tier0_d['Product'])}\n"
        f"{h_mod} : {{ TypeName: 'DupireLocalVolatilityModel', MarketDataSet: &{h_mds}, FinancingCurveIdentifiers: ['RUB_RUONIA_OIS'] }}\n"
        f"{h_mds} : {repr(tier0_d['Model']['MarketDataSet'])}\n"
        f"Result : {{ ProductValue: 0.25, CashFlows: [{{ Description: 'Option', Date: '2027-02-09T00:00:00Z', Amount: -0.1 }}] }}\n",
        encoding="utf-8",
    )
    parsed_file_req, parsed_file_res = parse_murex_autocall_dag(dag_file)
    assert parsed_file_req.Queries == [AutocallQueryMetric.ProductValue]
    assert parsed_file_res["Option"] == pytest.approx(-0.1)

    # SCH-008: get_coupon_rates(apply_dcc=True) with FixedRate
    prod_rate = copy.deepcopy(base_d["Product"])
    prod_rate["FixedCouponAmount"] = None
    prod_rate["FixedRate"] = 0.16
    prod_rate["DayCountConvention"] = "Act360"
    spec_rate = AutoCallableProductSpec.model_validate(prod_rate)
    scaled_rates = spec_rate.get_coupon_rates(apply_dcc=True)
    assert len(scaled_rates) == 12
    assert all(0.0 < x < 0.16 for x in scaled_rates)

    # SCH-019 & EXT-003: strict Murex validators
    with pytest.raises(ValidationError):
        RatesScheduleSpec(Dates=[], Rates=[])
    with pytest.raises(ValidationError):
        RatesScheduleSpec(Dates=[datetime.date(2027, 1, 1)], Rates=[0.1, 0.2])
    with pytest.raises(ValidationError):
        RatesScheduleSpec(Dates=[datetime.date(2027, 1, 1)], Rates=[], Spreads=[0.1, 0.2])
    with pytest.raises(ValidationError):
        RatesScheduleSpec(Dates=[datetime.date(2027, 1, 1)], Rates=[], Spreads=None)
    with pytest.raises(ValidationError):
        RepoScheduleSpec(Periods=["1Y"], Rates=[0.1, 0.2])
    with pytest.raises(ValidationError):
        RepoScheduleSpec(Periods=[], Rates=[])
    with pytest.raises(ValidationError):
        RepoScheduleSpec(Periods=["BAD_PERIOD"], Rates=[0.1])

    # MarketDataSetSpec validators: empty Spots / <=0 Spots / empty Vol / empty Curves
    bad_mds = copy.deepcopy(base_d["Model"]["MarketDataSet"])
    bad_mds["Spots"]["MAGN_RUB"] = 0.0
    with pytest.raises(ValidationError, match="strictly positive"):
        MarketDataSetSpec.model_validate(bad_mds)
    bad_mds["Spots"]["MAGN_RUB"] = -5.0
    with pytest.raises(ValidationError, match="strictly positive"):
        MarketDataSetSpec.model_validate(bad_mds)
    bad_mds["Spots"] = {}
    with pytest.raises(ValidationError):
        MarketDataSetSpec.model_validate(bad_mds)
    bad_mds2 = copy.deepcopy(base_d["Model"]["MarketDataSet"])
    bad_mds2["RatesCurvesBundles"] = {}
    with pytest.raises(ValidationError):
        MarketDataSetSpec.model_validate(bad_mds2)
    bad_mds3 = copy.deepcopy(base_d["Model"]["MarketDataSet"])
    bad_mds3["VolatilitySurfaces"] = {}
    with pytest.raises(ValidationError):
        MarketDataSetSpec.model_validate(bad_mds3)

    # AutoCallableProductSpec validators: duplicate IDs, pair index != 2, negative strike/gearing/KO, LowerStrike > Strike
    bad_p = copy.deepcopy(base_d["Product"])
    bad_p["UnderlyingIdentifiers"][1] = bad_p["UnderlyingIdentifiers"][0]
    with pytest.raises(ValidationError, match="duplicates"):
        AutoCallableProductSpec.model_validate(bad_p)

    bad_pair = copy.deepcopy(base_d["Product"])
    bad_pair["CallabilityMonitoringIndex"] = "RatioAB"
    with pytest.raises(ValidationError, match="requires exactly 2 underlyings"):
        AutoCallableProductSpec.model_validate(bad_pair)

    for field, val in (
        ("Strike", -0.1),
        ("LowerStrike", 1.2),
        ("OptionGearing", -1.0),
        ("AutoCallTriggerLevels", [-1.0] * 12),
        ("UnderlyingReferenceSpots", [0.0] * 5),
        ("FixedCouponAmount", [-0.01] * 12),
        ("CouponBarrierType", "InvalidType"),
        ("CouponBarrierDirection", "Sideways"),
    ):
        bp = copy.deepcopy(base_d["Product"])
        bp[field] = val
        with pytest.raises(ValidationError):
            AutoCallableProductSpec.model_validate(bp)

    # CashFlowRow aliases (PaymentDate, ExpectedValue, PaymentValue)
    cf_a = CashFlowRow.model_validate(
        {"Description": "Coupon", "PaymentDate": "2027-02-09T00:00:00Z", "ExpectedValue": 0.05}
    )
    cf_b = CashFlowRow.model_validate(
        {"Description": "Notional", "Date": datetime.date(2027, 2, 9), "PaymentValue": 1.0}
    )
    assert cf_a.Amount == pytest.approx(0.05)
    assert cf_b.PaymentValue == pytest.approx(1.0)


def test_sch_additional_dag_datetime_and_smile_validation_branches(tmp_path: Path) -> None:
    from deepvol.api.schemas_eq_autocall import (
        EquityVolSmileCollectionSpec,
        FXVolatilitySmileCollectionSpec,
        _parse_datetime_literal,
    )

    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    base_d = req.model_dump()
    base_d.pop("stripped_legacy_fields", None)

    # _parse_datetime_literal Murex Python literal forms, tzinfo, microsecond, and error paths
    dt_z = _parse_datetime_literal("datetime.datetime(2027, 2, 9, 12, 30, 45, tzinfo=datetime.timezone.utc)")
    assert dt_z.tzinfo == datetime.timezone.utc and dt_z.hour == 12
    dt_us = _parse_datetime_literal(
        "datetime.datetime(2027, 2, 9, 12, 30, 45, microsecond=123456, tzinfo=datetime.timezone(datetime.timedelta(hours=3)))"
    )
    assert dt_us.microsecond == 123456
    dt_date = _parse_datetime_literal("datetime.date(2027, 2, 9)")
    assert dt_date.year == 2027
    with pytest.raises(ValueError):
        _parse_datetime_literal("None")
    with pytest.raises(ValueError):
        _parse_datetime_literal("datetime.datetime(bad)")

    # parse_murex_autocall_dag direct request dict + {sha1: node_dict} with "Root" key + errors
    direct_d = copy.deepcopy(base_d)
    direct_d["Result"] = {"ProductValue": 0.88}
    r_direct, res_direct = parse_murex_autocall_dag(direct_d)
    assert r_direct.TradeInfo == req.TradeInfo and res_direct["ProductValue"] == pytest.approx(0.88)

    h_root = "f" * 40
    h_prod = "e" * 40
    h_mod = "d" * 40
    dag_with_root = {
        "Root": f"@{h_root}",
        h_root: {
            "Product": f"&{h_prod}",
            "Model": f"!{h_mod}",
            "Queries": ["ProductValue"],
        },
        h_prod: copy.deepcopy(base_d["Product"]),
        h_mod: copy.deepcopy(base_d["Model"]),
    }
    r_root, _ = parse_murex_autocall_dag(dag_with_root)
    assert len(r_root.Product.ObservationDates) == 12

    with pytest.raises(ValueError, match="root trade request node"):
        parse_murex_autocall_dag({"Root": "@" + "0" * 40, "a" * 40: {"Foo": 1}})
    with pytest.raises(TypeError):
        parse_murex_autocall_dag(42)  # type: ignore[arg-type]

    # AutoCallableProductSpec additional length & field validators
    bp_dates = copy.deepcopy(base_d["Product"])
    bp_dates["PaymentDates"] = bp_dates["PaymentDates"][:-1]
    with pytest.raises(ValidationError):
        AutoCallableProductSpec.model_validate(bp_dates)

    bp_ref = copy.deepcopy(base_d["Product"])
    bp_ref["UnderlyingReferenceSpots"] = bp_ref["UnderlyingReferenceSpots"][:-1]
    with pytest.raises(ValidationError):
        AutoCallableProductSpec.model_validate(bp_ref)

    bp_list = copy.deepcopy(base_d["Product"])
    bp_list["FixedCouponAmount"] = [0.05] * 3
    with pytest.raises(ValidationError):
        AutoCallableProductSpec.model_validate(bp_list)

    spec_list_rate = copy.deepcopy(base_d["Product"])
    spec_list_rate["FixedCouponAmount"] = None
    spec_list_rate["FixedRate"] = [0.12] * 12
    assert len(AutoCallableProductSpec.model_validate(spec_list_rate).get_coupon_rates(apply_dcc=True)) == 12

    # Smile validation errors
    vkco_d = copy.deepcopy(base_d["Model"]["MarketDataSet"]["VolatilitySurfaces"]["VKCO_RUB"])
    vkco_d["VolatilityQuotes"][0] = vkco_d["VolatilityQuotes"][0][:-1]
    with pytest.raises(ValidationError):
        EquityVolSmileCollectionSpec.model_validate(vkco_d)

    fx_bad = {
        "Identifier": "USD/RUB",
        "TypeName": "FXVolatilitySmileCollection",
        "DayCountConvention": "Act365",
        "Deltas": ["-25D", "ATM"],
        "Tenors": ["1M", "6M"],
        "VolatilityQuotes": [[0.20, 0.21]],
    }
    with pytest.raises(ValidationError):
        FXVolatilitySmileCollectionSpec.model_validate(fx_bad)

    # Dividend validators + cyclical DAG & missing filename path resolution
    from deepvol.api.schemas_eq_autocall import DividendItemSpec, PredefinedDividendsSpec

    dt_ex = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    dt_pay = datetime.datetime(2026, 6, 5, tzinfo=datetime.timezone.utc)
    with pytest.raises(ValidationError):
        DividendItemSpec(ExDate=dt_ex, PaymentDate=dt_pay, Amount=-1.0)
    for bad_amt in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValidationError):
            DividendItemSpec(ExDate=dt_ex, PaymentDate=dt_pay, Amount=bad_amt)
    with pytest.raises(ValidationError):
        PredefinedDividendsSpec(ExDates=[datetime.date(2026, 6, 1)], Amounts=[-1.0])
    with pytest.raises(ValidationError):
        PredefinedDividendsSpec(ExDates=[datetime.date(2026, 6, 1)], Amounts=[1.0, 2.0])

    h_cyc_a = "1" * 40
    h_cyc_b = "2" * 40
    cyc_file = tmp_path / "cyclic_dag.txt"
    cyc_file.write_text(
        f"@{h_cyc_a}\n"
        f"{h_cyc_a} : {{ Product: &{h_cyc_b}, Model: &{h_cyc_b} }}\n"
        f"{h_cyc_b} : {{ Product: &{h_cyc_a}, Model: &{h_cyc_a} }}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cycle"):
        parse_murex_autocall_dag(cyc_file)

    r_rel, _ = parse_murex_autocall_dag("eq_autocallable_request.txt", allow_file_path=True)
    assert r_rel.TradeInfo == req.TradeInfo

    # DAG tokenizer/scalar/stream helpers & anonymous dict + root errors
    from deepvol.api.schemas_eq_autocall import _DAGTokenStream, _parse_scalar_token, _tokenize_dag_fast

    ts = _DAGTokenStream("{ a: 'x' }")
    assert ts.peek() == ("LBRACE", "{")
    assert ts.expect("LBRACE") == "{"
    with pytest.raises(ValueError):
        ts.expect("RBRACE")
    assert len(_tokenize_dag_fast("{ a: 1 }")) >= 4
    assert _parse_scalar_token("<skipped>") is None
    assert _parse_scalar_token("true") is True
    assert _parse_scalar_token("false") is False
    assert _parse_scalar_token("null") is None
    assert _parse_scalar_token("1e-4") == pytest.approx(1e-4)
    assert _parse_scalar_token("a" * 40) == "a" * 40
    with pytest.raises(ValueError):
        _parse_scalar_token("}")

    json_d = req.model_dump(mode="json")
    json_d.pop("stripped_legacy_fields", None)
    anon_text = f"{{ Product: {repr(json_d['Product'])}, Model: {repr(json_d['Model'])}, Queries: ['ProductValue'] }}"
    r_anon, _ = parse_murex_autocall_dag(anon_text)
    assert len(r_anon.Product.ObservationDates) == 12
    with pytest.raises((ValueError, KeyError)):
        parse_murex_autocall_dag(f"@{'a' * 40}\n{'b' * 40} : {{ Foo: 1 }}")
    with pytest.raises(ValueError):
        parse_murex_autocall_dag("   ")
    with pytest.raises(ValueError):
        parse_murex_autocall_dag("BOGUS_TOKEN")

    # _parse_str_dag_at & error paths for unclosed literals, unexpected delimiters, missing colon
    from deepvol.api.schemas_eq_autocall import _FAST_TOK_RE, _parse_str_dag_at

    toks_dt = _FAST_TOK_RE.findall(
        "{ a: datetime.datetime(2027, 2, 9, 0, 0, 15, tzinfo=datetime.timezone.utc), "
        "b: datetime.date(2027, 2, 9), c: <skipped>, d: True, e: False, f: None, "
        f"g: {'a' * 40}, h: 123abc }}"
    )
    res_map, _ = _parse_str_dag_at(toks_dt, 0, len(toks_dt))
    assert res_map["c"] is None and res_map["d"] is True and res_map["e"] is False and res_map["f"] is None

    for bad_dag_str in (
        "{ a: }",
        "}",
        "]",
        "{ : 1 }",
        "{ a 1 }",
        "{ a: : }",
        "{ 'unclosed: 1 }",
        "{ a: 'unclosed }",
        "{ a: ",
    ):
        bad_toks = _FAST_TOK_RE.findall(bad_dag_str)
        with pytest.raises(ValueError):
            _parse_str_dag_at(bad_toks, 0, len(bad_toks))

    # Additional tokenizer/parser branches (_DAGTokenStream, _parse_dag_at, _extract_reference_hash)
    from deepvol.api.schemas_eq_autocall import (
        _extract_reference_hash,
        _parse_dag_at,
        _parse_dag_value,
    )

    with pytest.raises(ValueError, match="Unclosed string literal"):
        _parse_scalar_token("'")
    ts_empty = _DAGTokenStream("")
    assert ts_empty.peek() is None
    with pytest.raises(ValueError):
        ts_empty.next()

    assert _parse_scalar_token("'single'") == "single"
    assert _parse_scalar_token("123abc") == "123abc"
    assert _parse_scalar_token("datetime.date(2026, 6, 1)") == datetime.datetime(
        2026, 6, 1, tzinfo=datetime.timezone.utc
    )
    assert _parse_scalar_token("@" + "a" * 40) == "@" + "a" * 40
    assert _parse_scalar_token("custom_ident") == "custom_ident"

    tok_ok = _tokenize_dag_fast(
        "{ 'quoted_k': [true, false, null, "
        + "a" * 40
        + ", &"
        + "b" * 40
        + ", datetime.date(2026, 6, 1), 'sq', bad123] }"
    )
    ts_ok = _DAGTokenStream.__new__(_DAGTokenStream)
    ts_ok._raw_tokens = tok_ok
    ts_ok.tokens = []
    ts_ok.idx = 0
    parsed_ok = _parse_dag_value(ts_ok)
    assert parsed_ok["quoted_k"][0] is True and parsed_ok["quoted_k"][1] is False and parsed_ok["quoted_k"][2] is None

    for bad_fast in ("", "{ [1]: 2 }", "{ a 2 }", "{ a: 1", "[ 1 2", "{ a: ] }"):
        bad_ftoks = _tokenize_dag_fast(bad_fast)
        with pytest.raises(ValueError):
            _parse_dag_at(bad_ftoks, 0, len(bad_ftoks))

    h_node = "c" * 40
    assert _extract_reference_hash(123, {h_node: {}}) is None
    assert _extract_reference_hash(h_node.upper(), {h_node: {}}) == h_node
    assert _extract_reference_hash("d" * 40, {h_node: {}}) is None


def test_schema_audit_blind_spots() -> None:
    for lfi_or_long in ("/etc/passwd", "x" * 260, "data/autocall/eq_autocallable_request.txt"):
        with pytest.raises(ValueError):
            parse_murex_autocall_dag(lfi_or_long)

    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    base_d = req.model_dump(mode="json")
    base_d.pop("stripped_legacy_fields", None)

    required_indices = {
        "EquallyWeightedBasket",
        "SecondWorstOf",
        "ThirdWorstOf",
        "SecondBestOf",
        "MultiPerf",
        "Lookback",
        "TwinWin",
        "Outperformance",
        "Spread",
        "Rainbow",
        "Ratio",
        "SpreadBestWorst",
        "SpreadBestAverage",
        "SpreadAverageWorst",
    }
    actual_indices = {m.value for m in MonitoringIndexEnum}
    assert required_indices.issubset(actual_indices)
    for idx_name in (
        "EquallyWeightedBasket",
        "SecondWorstOf",
        "ThirdWorstOf",
        "SecondBestOf",
        "MultiPerf",
    ):
        d_idx = copy.deepcopy(base_d)
        d_idx["Product"]["CallabilityMonitoringIndex"] = idx_name
        d_idx["Product"]["CouponBarrierMonitoringIndex"] = idx_name
        d_idx["Product"]["OptionMonitoringIndex"] = idx_name
        parsed_idx = EQAutocallableTradeRequest.model_validate(d_idx)
        assert parsed_idx.Product.CallabilityMonitoringIndex.value == idx_name

    # Currency must survive serialization on dividends and vol smiles.
    d_ccy = copy.deepcopy(base_d)
    first_tkr = d_ccy["Product"]["UnderlyingIdentifiers"][0]
    d_ccy["Model"]["MarketDataSet"]["VolatilitySurfaces"][first_tkr]["Currency"] = "USD"
    if first_tkr in d_ccy["Model"]["MarketDataSet"].get("Dividends", {}):
        d_ccy["Model"]["MarketDataSet"]["Dividends"][first_tkr]["Currency"] = "USD"
    d_ccy["Product"]["PredefinedDividends"] = {
        first_tkr: {
            "Currency": "USD",
            "ExDates": ["2026-12-01"],
            "Amounts": [1.5],
        }
    }
    req_ccy = EQAutocallableTradeRequest.model_validate(d_ccy)
    dumped_ccy = req_ccy.model_dump(mode="json")
    assert dumped_ccy["Model"]["MarketDataSet"]["VolatilitySurfaces"][first_tkr]["Currency"] == "USD"
    assert dumped_ccy["Product"]["PredefinedDividends"][first_tkr]["Currency"] == "USD"
    if first_tkr in dumped_ccy["Model"]["MarketDataSet"].get("Dividends", {}):
        assert dumped_ccy["Model"]["MarketDataSet"]["Dividends"][first_tkr]["Currency"] == "USD"

    # Explicit None values on optional top-level/nested schema fields must coalesce cleanly.
    d_none_1d = copy.deepcopy(base_d)
    keep_tkr = first_tkr
    d_none_1d["Product"]["UnderlyingIdentifiers"] = [keep_tkr]
    d_none_1d["Product"]["UnderlyingReferenceSpots"] = [d_none_1d["Product"]["UnderlyingReferenceSpots"][0]]
    d_none_1d["Product"]["DayCountConvention"] = None
    d_none_1d["Product"]["CallabilityMonitoringIndex"] = None
    d_none_1d["Product"]["PredefinedDividends"] = None
    d_none_1d["Model"]["TypeName"] = None
    d_none_1d["Model"]["FinancingCurveIdentifiers"] = None
    d_none_1d["Model"]["MarketDataSet"]["Spots"] = {keep_tkr: d_none_1d["Model"]["MarketDataSet"]["Spots"][keep_tkr]}
    d_none_1d["Model"]["MarketDataSet"]["VolatilitySurfaces"] = {
        keep_tkr: d_none_1d["Model"]["MarketDataSet"]["VolatilitySurfaces"][keep_tkr]
    }
    d_none_1d["Model"]["MarketDataSet"]["Correlations"] = None
    d_none_1d["Model"]["MarketDataSet"]["RepoSpreadCurves"] = None
    d_none_1d["Model"]["MarketDataSet"]["Dividends"] = None
    d_none_1d["Model"]["MarketDataSet"]["Fixings"] = None
    d_none_1d["stripped_legacy_fields"] = None
    req_none = EQAutocallableTradeRequest.model_validate(d_none_1d)
    assert req_none.Product.DayCountConvention == DayCountConventionEnum.Act360
    assert req_none.Model.TypeName == "DupireLocalVolatilityModel"

    # Ticker-keyed Fixings without Identifier on Pydantic schema + {date_str: float} dict fixings on collapser.
    from deepvol.models.autocall.seasoned_collapser import collapse_seasoned_schedule

    d_fix = copy.deepcopy(base_d)
    obs_iso = str(d_fix["Product"]["ObservationDates"][0])
    d_fix["Model"]["MarketDataSet"]["Fixings"] = {
        keep_tkr: {"Dates": [obs_iso], "Values": [99.5]},
    }
    req_fix = EQAutocallableTradeRequest.model_validate(d_fix)
    assert req_fix.Model.MarketDataSet.Fixings[keep_tkr].Identifier == keep_tkr

    val_fix = datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc)
    cs_dict_fix = collapse_seasoned_schedule(
        as_of_date=val_fix,
        start_date=val_fix - datetime.timedelta(days=60),
        observation_dates=[val_fix - datetime.timedelta(days=10), val_fix + datetime.timedelta(days=30)],
        payment_dates=[val_fix - datetime.timedelta(days=7), val_fix + datetime.timedelta(days=33)],
        underlying_identifiers=[keep_tkr],
        underlying_reference_spots=[100.0],
        spots_t0=[90.0],
        ko_barriers=[0.95, 1.15],
        cpn_barriers=[0.80, 0.80],
        fixed_coupon_amount=0.04,
        fixings={keep_tkr: {(val_fix - datetime.timedelta(days=10)).date().isoformat(): 99.5}},
    )
    assert cs_dict_fix.already_autocalled is True

    from typing import Callable

    # Non-finite values must be rejected across request and response schemas.
    bad_mutations: list[Callable[[dict[str, Any]], Any]] = [
        lambda x: x["Product"].__setitem__("Strike", float("nan")),
        lambda x: x["Product"].__setitem__("OptionGearing", float("inf")),
        lambda x: x["Product"]["AutoCallTriggerLevels"].__setitem__(0, float("nan")),
        lambda x: x["Model"]["MarketDataSet"]["Spots"].__setitem__(keep_tkr, float("nan")),
        lambda x: next(iter(x["Model"]["MarketDataSet"]["RatesCurvesBundles"].values()))["RatesSchedule"][
            "Rates"
        ].__setitem__(0, float("inf")),
        lambda x: x["Model"]["MarketDataSet"]["VolatilitySurfaces"][keep_tkr]["VolatilityQuotes"][0].__setitem__(
            0, float("nan")
        ),
        lambda x: x["Model"]["MarketDataSet"]["Fixings"][keep_tkr]["Values"].__setitem__(0, float("nan")),
        lambda x: x["Model"]["MarketDataSet"]["Correlations"].__setitem__(
            next(iter(x["Model"]["MarketDataSet"]["Correlations"])), float("inf")
        ),
    ]
    for mutate in bad_mutations:
        bad_d = copy.deepcopy(base_d)
        mutate(bad_d)
        with pytest.raises(ValidationError):
            EQAutocallableTradeRequest.model_validate(bad_d)

    with pytest.raises(ValidationError):
        CashFlowRow(
            Description="Option",
            Leg=0,
            Date=datetime.datetime(2027, 2, 9, tzinfo=datetime.timezone.utc),
            StatusText="Estimate",
            Amount=float("nan"),
        )
    with pytest.raises(ValidationError):
        EQAutocallableTradeResponse(ProductValue=float("inf"), ReportingCurrency="RUB")

    cf_intransit = CashFlowRow.model_validate(
        {
            "Description": "FundingLeg",
            "Leg": 0,
            "Date": "2026-10-30T00:00:00Z",
            "StatusText": "In-Transit",
            "Amount": -0.01,
            "ExpectedValue": None,
        }
    )
    assert cf_intransit.Amount == pytest.approx(-0.01)

    resp_2d = EQAutocallableTradeResponse.model_validate(
        {
            "ProductValue": 0.5,
            "ReportingCurrency": "RUB",
            "Gamma": {"A": {"A": 0.1, "B": -0.02}, "B": {"A": -0.02, "B": 0.1}},
            "Cega": {"A": {"B": 0.03}, "B": {"A": 0.03}},
        }
    )
    gamma_map: Any = resp_2d.Gamma
    cega_map: Any = resp_2d.Cega
    assert isinstance(gamma_map, dict) and float(gamma_map["A"]["B"]) == pytest.approx(-0.02)
    assert isinstance(cega_map, dict) and float(cega_map["A"]["B"]) == pytest.approx(0.03)


def test_canonical_tier3_config_and_to_decimal_merged_rho() -> None:
    """Verifies `CanonicalTier3Config` resolver invariants and `to_decimal_merged_rho` scaling/merging."""
    from types import SimpleNamespace
    from deepvol.api.schemas_eq_autocall import to_decimal_merged_rho
    from deepvol.models.autocall.feature_layout import (
        CanonicalTier3Config,
        resolve_canonical_tier3_config,
    )

    cfg_cuda = resolve_canonical_tier3_config(6, 48, device_type="cuda")
    assert isinstance(cfg_cuda, CanonicalTier3Config)
    assert cfg_cuda == CanonicalTier3Config(
        n_paths=4096,
        n_replicates=4,
        n_substeps=4,
        lv_grid_nt=32,
        lv_grid_ny=64,
        cv_mode="multifidelity_gbm",
        n_cheap_mf=262144,
        seed=42,
        adaptive_substeps=False,
    )

    cfg_cuda_mid = resolve_canonical_tier3_config(6, 20, device_type="cuda")
    assert cfg_cuda_mid == CanonicalTier3Config(
        n_paths=24576,
        n_replicates=24,
        n_substeps=4,
        lv_grid_nt=32,
        lv_grid_ny=64,
        cv_mode="multifidelity_gbm",
        n_cheap_mf=262144,
        seed=42,
        adaptive_substeps=False,
    )

    cfg_cpu = resolve_canonical_tier3_config(5, 12, device_type="cpu")
    assert cfg_cpu == CanonicalTier3Config(
        n_paths=4096,
        n_replicates=4,
        n_substeps=4,
        lv_grid_nt=32,
        lv_grid_ny=64,
        cv_mode="multifidelity_gbm",
        n_cheap_mf=32768,
        seed=42,
        adaptive_substeps=False,
    )

    cfg_override = resolve_canonical_tier3_config(3, 24, device_type="cpu", n_fallback_paths=16384, seed=7)
    assert cfg_override.n_paths == 16384
    assert cfg_override.n_replicates == 8
    assert cfg_override.n_cheap_mf == 131072
    assert cfg_override.seed == 7
    assert cfg_override.adaptive_substeps is False

    assert (resolve_canonical_tier3_config(2, 24, device_type="cuda").n_paths, resolve_canonical_tier3_config(2, 24, device_type="cuda").n_replicates) == (24576, 24)
    assert (resolve_canonical_tier3_config(2, 28, device_type="cuda").n_paths, resolve_canonical_tier3_config(2, 28, device_type="cuda").n_replicates) == (8192, 8)
    assert (resolve_canonical_tier3_config(3, 20, device_type="cuda").n_paths, resolve_canonical_tier3_config(3, 20, device_type="cuda").n_replicates) == (12288, 12)
    assert (resolve_canonical_tier3_config(3, 22, device_type="cuda").n_paths, resolve_canonical_tier3_config(3, 22, device_type="cuda").n_replicates) == (8192, 8)
    assert (resolve_canonical_tier3_config(3, 38, device_type="cuda").n_paths, resolve_canonical_tier3_config(3, 38, device_type="cuda").n_replicates) == (8192, 8)
    assert (resolve_canonical_tier3_config(5, 6, device_type="cuda").n_paths, resolve_canonical_tier3_config(5, 6, device_type="cuda").n_replicates) == (12288, 12)
    assert (resolve_canonical_tier3_config(5, 13, device_type="cuda").n_paths, resolve_canonical_tier3_config(5, 13, device_type="cuda").n_replicates) == (8192, 8)
    assert (resolve_canonical_tier3_config(4, 23, device_type="cuda").n_paths, resolve_canonical_tier3_config(4, 23, device_type="cuda").n_replicates) == (32768, 32)
    assert (resolve_canonical_tier3_config(4, 35, device_type="cuda").n_paths, resolve_canonical_tier3_config(4, 35, device_type="cuda").n_replicates) == (12288, 12)
    assert (resolve_canonical_tier3_config(6, 29, device_type="cuda").n_paths, resolve_canonical_tier3_config(6, 29, device_type="cuda").n_replicates) == (16384, 16)
    assert (resolve_canonical_tier3_config(5, 8, device_type="cuda").n_paths, resolve_canonical_tier3_config(5, 8, device_type="cuda").n_replicates) == (16384, 16)
    assert (resolve_canonical_tier3_config(2, 6, device_type="cuda").n_paths, resolve_canonical_tier3_config(2, 6, device_type="cuda").n_replicates) == (32768, 32)
    assert (resolve_canonical_tier3_config(2, 23, device_type="cuda").n_paths, resolve_canonical_tier3_config(2, 23, device_type="cuda").n_replicates) == (24576, 24)
    assert (resolve_canonical_tier3_config(6, 36, device_type="cuda").n_paths, resolve_canonical_tier3_config(6, 36, device_type="cuda").n_replicates) == (4096, 4)
    assert (resolve_canonical_tier3_config(6, 37, device_type="cuda").n_paths, resolve_canonical_tier3_config(6, 37, device_type="cuda").n_replicates) == (4096, 4)
    assert (resolve_canonical_tier3_config(6, 11, device_type="cpu").n_paths, resolve_canonical_tier3_config(6, 11, device_type="cpu").n_replicates) == (4096, 4)
    cfg_cpu_long = resolve_canonical_tier3_config(3, 40, device_type="cpu")
    assert (cfg_cpu_long.n_paths, cfg_cpu_long.n_replicates, cfg_cpu_long.cv_mode) == (4096, 8, "multifidelity_gbm")
    cfg_cpu_d6_long = resolve_canonical_tier3_config(6, 39, device_type="cpu")
    assert (cfg_cpu_d6_long.n_paths, cfg_cpu_d6_long.n_replicates, cfg_cpu_d6_long.cv_mode) == (1024, 4, "multifidelity_gbm")

    # Quote -> decimal conversion with mixed Percentage / Decimal bundles + RepoRho merging
    fake_req = SimpleNamespace(
        Model=SimpleNamespace(
            MarketDataSet=SimpleNamespace(
                RatesCurvesBundles={
                    "RUB_RUONIA_OIS": SimpleNamespace(RateQuoteExpression="Percentage"),
                    "USD_SOFR": SimpleNamespace(RateQuoteExpression="Decimal"),
                },
                RepoSpreadCurves={
                    "SBER_RUB": SimpleNamespace(RateQuoteExpression="Percentage"),
                },
            )
        )
    )
    resp_quote = {
        "Rho": {"RUB_RUONIA_OIS": [-0.02, -0.01], "USD_SOFR": [-0.5]},
        "RepoRho": {"SBER_RUB": [0.004, 0.006]},
        "rho_unit_convention": "quote",
    }
    merged_q = to_decimal_merged_rho(resp_quote, request=fake_req)
    assert merged_q["RUB_RUONIA_OIS"] == pytest.approx([-2.0, -1.0])
    assert merged_q["USD_SOFR"] == pytest.approx([-0.5])
    assert merged_q["SBER_RUB"] == pytest.approx([0.4, 0.6])

    resp_dec = {
        "Rho": {"RUB_RUONIA_OIS": [-2.0, -1.0]},
        "RepoRho": {"SBER_RUB": [0.4, 0.6]},
        "rho_unit_convention": "decimal",
    }
    merged_d = to_decimal_merged_rho(resp_dec, request=fake_req)
    assert merged_d["RUB_RUONIA_OIS"] == pytest.approx([-2.0, -1.0])
    assert merged_d["SBER_RUB"] == pytest.approx([0.4, 0.6])







