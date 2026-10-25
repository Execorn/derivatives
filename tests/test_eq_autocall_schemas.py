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

REQUEST_TXT_PATH = Path("/home/execorn/programming/derivatives/eq_autocallable_request.txt")


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
    """Verifies 0-error parsing of `/home/execorn/programming/derivatives/eq_autocallable_request.txt`."""
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
    req_from_str_path, _ = parse_murex_autocall_dag(str(REQUEST_TXT_PATH))
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
