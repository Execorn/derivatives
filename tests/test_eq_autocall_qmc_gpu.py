from __future__ import annotations

import copy
import datetime
import math
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
import pytest
from scipy.special import ndtr
import torch

from deepvol.api.schemas_eq_autocall import (
    CurveInterpolationMethodEnum,
    EQAutocallableTradeRequest,
    EquityVolSmileCollectionSpec,
    MonitoringIndexEnum,
    PredefinedDividendsSpec,
    parse_murex_autocall_dag,
)
from deepvol.models.autocall import (
    AutocallPreprocState,
    preprocess_eq_autocall,
    price_eq_autocall_dupire_qmc,
)
from deepvol.models.autocall.dupire_qmc_gpu import (
    CpuNumpyDupireQMC,
    DupireQMCConfig,
    DupireQMCGpuSolver,
    _generate_gpu_sobol_tile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REF_REQUEST_PATH = (
    REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt"
    if (REPO_ROOT / "data" / "autocall" / "eq_autocallable_request.txt").exists()
    else Path("/home/execorn/programming/derivatives/data/autocall/eq_autocallable_request.txt")
)


@pytest.fixture(scope="module")
def parsed_dag() -> tuple[EQAutocallableTradeRequest, dict[str, Any]]:
    return parse_murex_autocall_dag(REF_REQUEST_PATH)


def _build_synthetic_request(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
    *,
    d_assets: int = 4,
    m_periods: int = 12,
    spot_scale: float = 1.0,
    div_amt: float = 0.0,
) -> EQAutocallableTradeRequest:
    req_orig, _ = parsed_dag
    req = copy.deepcopy(req_orig)
    as_of = datetime.datetime(2026, 10, 23, 0, 0, tzinfo=datetime.timezone.utc)
    req.Product.StartDate = as_of
    obs_dates = [as_of + datetime.timedelta(days=30 * (m + 1)) for m in range(m_periods)]
    pay_dates = [d + datetime.timedelta(days=5) for d in obs_dates]
    req.Product.ObservationDates = obs_dates
    req.Product.PaymentDates = pay_dates
    req.Product.AutoCallTriggerLevels = [1.05] * max(m_periods - 1, 0) + [500.0]
    req.Product.CouponBarrierLevels = [0.75] * m_periods
    req.Product.FixedCouponAmount = 0.025
    req.Product.FixedRate = None
    req.Product.IsMemory = True
    req.Product.Strike = 0.90
    req.Product.OptionGearing = 1.11111
    req.Product.OptionDownAndInBarrierLevel = 0.70

    tickers = [f"EQ{i:02d}_RUB" for i in range(d_assets)]
    req.Product.UnderlyingIdentifiers = tickers
    req.Product.UnderlyingReferenceSpots = [100.0] * d_assets
    req.Product.PredefinedDividends = (
        {
            tickers[0]: PredefinedDividendsSpec(
                ExDates=[as_of + datetime.timedelta(days=45)],
                Amounts=[div_amt],
            )
        }
        if div_amt > 0.0
        else None
    )

    mds = req.Model.MarketDataSet
    mds.AsOfDate = as_of
    mds.Spots = {t: 100.0 * spot_scale for t in tickers}
    mds.Dividends = {}
    mds.Fixings = {}
    mds.RepoSpreadCurves = {}
    mds.Correlations = {
        f"{tickers[i]}&{tickers[j]}": 0.45
        for i in range(d_assets)
        for j in range(i + 1, d_assets)
    }
    disc = mds.RatesCurvesBundles["RUB_RUONIA_OIS"]
    disc.AsOfDate = as_of
    disc.InterpolationMethod = CurveInterpolationMethodEnum.Linear
    disc.RatesSchedule.Dates = [
        as_of + datetime.timedelta(days=30),
        as_of + datetime.timedelta(days=365),
        as_of + datetime.timedelta(days=365 * 20),
    ]
    disc.RatesSchedule.Rates = [10.0, 10.0, 10.0]

    exp_strs = [
        (as_of + datetime.timedelta(days=90)).isoformat(),
        (as_of + datetime.timedelta(days=365)).isoformat(),
        (as_of + datetime.timedelta(days=365 * 20)).isoformat(),
    ]
    mds.VolatilitySurfaces = {
        t: EquityVolSmileCollectionSpec.model_validate(
            {
                "Identifier": t,
                "DayCountConvention": "Act365",
                "ExpirationDates": exp_strs,
                "Strikes": [50.0, 80.0, 100.0, 120.0, 150.0],
                "VolatilityQuotes": [
                    [0.27, 0.26, 0.255, 0.255, 0.26],
                    [0.28, 0.27, 0.265, 0.265, 0.27],
                    [0.29, 0.28, 0.275, 0.275, 0.28],
                ],
            }
        )
        for t in tickers
    }
    return req


def test_cpu_numpy_dupire_qmc_facade_and_empty_training_batch(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    req_model, _ = parsed_dag
    state = preprocess_eq_autocall(req_model)

    cpu_solver = DupireQMCGpuSolver(device="cpu", backend="cpu")
    assert cpu_solver.backend_name == "CpuNumpyDupireQMC"
    res = cpu_solver.simulate_and_price(
        state,
        DupireQMCConfig(n_paths=1024, n_replicates=4, n_substeps=2, cv_mode="none"),
    )
    assert np.isfinite(res.product_value)
    assert res.standard_error > 0.0

    pv0, p0, q0, l0, d0 = CpuNumpyDupireQMC().simulate_batch_for_training([], compute_pathwise_delta=True)
    assert pv0.shape == (0,) and p0.shape == (0, 48) and q0.shape == (0, 48) and l0.shape == (0,)
    assert d0 is not None and d0.shape == (0, 1)

    pv_b, p_b, q_b, l_b, d_b = cpu_solver.simulate_batch_for_training(
        [state],
        config=DupireQMCConfig(n_paths=256, n_replicates=4, n_substeps=2),
        compute_pathwise_delta=True,
    )
    assert pv_b.dtype == torch.float64 and pv_b.shape == (1,)
    assert p_b.shape == (1, 48) and q_b.shape == (1, 48) and l_b.shape == (1,)
    assert d_b is not None and d_b.shape == (1, 8)

    # Verify all monitoring functionals and pathwise subgradients in dupire_qmc_gpu
    from deepvol.models.autocall.dupire_qmc_gpu import (
        _apply_loo_control_variate,
        _digital_ramp_torch,
        _eval_index_and_grad_gpu,
        _evaluate_monitoring_index_gpu,
    )

    perf3 = torch.tensor([[1.1, 0.9, 1.3], [1.0, 1.0, 1.0]], dtype=torch.float64)
    perm3 = torch.tensor([2, 0, 1], dtype=torch.int64)
    perf1 = torch.tensor([[1.05]], dtype=torch.float64)
    all_idx = (
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
        "UnknownCustom",
    )
    for idx_name in all_idx:
        w3 = _evaluate_monitoring_index_gpu(perf3, idx_name, perm3, asset_dim=-1)
        w3g, g3 = _eval_index_and_grad_gpu(perf3, idx_name, perm3, compute_grad=True, asset_dim=-1)
        w3n, g3n = _eval_index_and_grad_gpu(perf3, idx_name, perm3, compute_grad=False, asset_dim=-1)
        torch.testing.assert_close(w3, w3g, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w3, w3n, atol=1e-5, rtol=1e-5)
        assert g3 is not None and torch.isfinite(g3).all() and g3n is None
        _ = _evaluate_monitoring_index_gpu(perf1, idx_name, None, asset_dim=-1)
        _, g1 = _eval_index_and_grad_gpu(perf1, idx_name, None, compute_grad=True, asset_dim=-1)
        assert g1 is not None and g1.shape == (1, 1)

    _ = _digital_ramp_torch(torch.tensor([0.8, 1.0], dtype=torch.float64), 0.9, 0.0)
    est1, se1 = _apply_loo_control_variate(
        torch.tensor([[1.0, 1.2, 0.9, 1.1]], dtype=torch.float64),
        torch.tensor([[0.5, 0.6, 0.4, 0.5]], dtype=torch.float64),
        0.5,
    )
    assert np.isfinite(est1) and se1 >= 0.0


@pytest.mark.gpu
def test_gpu_dupire_qmc_reference_parity_cv_modes_and_sobol_warnings(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for GPU Dupire QMC verification")

    req_model, expected_res = parsed_dag
    ref_pv = float(expected_res["ProductValue"])
    ref_se = float(expected_res["StandardError"])
    state = preprocess_eq_autocall(req_model)

    res_cpu = price_eq_autocall_dupire_qmc(state, n_paths=32768, n_replicates=8, mode="dupire_lv", seed=42)

    for backend_mode in ("triton", "torch"):
        solver = DupireQMCGpuSolver(device="cuda", backend=backend_mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            for cv_mode in ("bs1d_exact", "multifidelity_gbm", "none"):
                cfg = DupireQMCConfig(
                    n_paths=32768,
                    n_replicates=8,
                    n_substeps=16,
                    mode="dupire_lv",
                    cv_mode=cv_mode,
                    seed=42,
                )
                res_gpu = solver.simulate_and_price(state, cfg)
                assert abs(res_gpu.product_value - ref_pv) <= max(1.5 * ref_se, 5e-4)
                assert abs(res_gpu.product_value - res_cpu.product_value) <= max(
                    5e-5, 2.5 * res_gpu.standard_error
                )
                assert res_gpu.standard_error > 0.0
                assert len(res_gpu.cashflows) == 37
                assert np.isfinite(res_gpu.l_put)
        sobol_warnings = [w for w in caught if "Sobol" in str(w.message)]
        assert len(sobol_warnings) == 0, f"Unexpected Sobol UserWarning: {sobol_warnings}"

    # Verify terminal_iv, fp64_audit, and m_rem == 0 paths on GPU
    res_tiv = solver.simulate_and_price(
        state,
        DupireQMCConfig(n_paths=4096, n_replicates=8, mode="terminal_iv", precision="fp64_audit", cv_mode="none"),
    )
    assert abs(res_tiv.product_value - ref_pv) <= 2e-3

    state_m0 = copy.copy(state)
    state_m0.m_rem = 0
    res_m0 = solver.simulate_and_price(state_m0)
    assert res_m0.standard_error == 0.0


@pytest.mark.gpu
def test_gpu_dupire_qmc_1d_dividend_puts_match_black76(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for GPU Dupire QMC verification")

    solver = DupireQMCGpuSolver(device="cuda", backend="auto")
    for div_amt in (0.0, 10.0, 20.0, 30.0):
        req = _build_synthetic_request(parsed_dag, d_assets=1, m_periods=1, div_amt=div_amt)
        req.Product.AutoCallTriggerLevels = [500.0]
        req.Product.CouponBarrierLevels = [500.0]
        req.Product.FixedCouponAmount = 0.0
        req.Product.Strike = 1.0
        req.Product.OptionGearing = 1.0
        req.Product.OptionDownAndInBarrierLevel = None
        req.Product.NoNotionalRepayment = True
        req.Product.AutoCallTriggerAbsoluteDigitalSpread = 0.0
        # Flatten vol surface for exact Black-76 benchmark
        for s_spec in req.Model.MarketDataSet.VolatilitySurfaces.values():
            assert isinstance(s_spec, EquityVolSmileCollectionSpec)
            s_spec.VolatilityQuotes = [[0.25] * 5 for _ in range(3)]
        state = preprocess_eq_autocall(req)
        res = solver.simulate_and_price(
            state,
            DupireQMCConfig(n_paths=32768, n_replicates=8, n_substeps=16, cv_mode="bs1d_exact"),
        )
        f_perf = float(state.forward_perf[0, 0].item())
        df = float(state.df_pay[0].item())
        v = float(state.v_perf[0, 0].item())
        d1 = (math.log(f_perf / 1.0) + 0.5 * v * v) / v
        d2 = d1 - v
        b76_pv = -df * (1.0 * float(ndtr(-d2)) - f_perf * float(ndtr(-d1)))
        assert abs(res.product_value - b76_pv) <= 5e-5, (
            f"div={div_amt}: GPU={res.product_value:.6f} vs B76={b76_pv:.6f}"
        )


@pytest.mark.gpu
def test_gpu_dupire_qmc_exotic_modifiers_memory_phoenix_and_philox_high_dim(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for GPU Dupire QMC verification")

    solver = DupireQMCGpuSolver(device="cuda", backend="auto")

    # 1. Multi-period Memory Phoenix with FundingLeg, Out/Down coupon, LowerStrike, OneStar, PartCall, RatioAB
    req_ex = _build_synthetic_request(parsed_dag, d_assets=2, m_periods=12, div_amt=3.0)
    req_ex.Product.UnderlyingIdentifiers = ["ZZ_RUB", "AA_RUB"]
    req_ex.Product.CallabilityMonitoringIndex = MonitoringIndexEnum.RatioAB
    req_ex.Product.CouponBarrierMonitoringIndex = MonitoringIndexEnum.DiffAB
    req_ex.Product.OptionMonitoringIndex = MonitoringIndexEnum.WorstOf
    req_ex.Product.CouponBarrierType = "Out"
    req_ex.Product.CouponBarrierDirection = "Down"
    req_ex.Product.LowerStrike = 0.50
    req_ex.Product.AutoCallCoupon = 0.03
    req_ex.Product.ForceCouponsOnAutocall = True
    req_ex.Product.OneStarStrike = 1.15
    req_ex.Product.OneStarCoupon = 0.04
    req_ex.Product.ParticipationCallStrike = 1.00
    req_ex.Product.ParticipationCallRate = 0.30
    req_ex.Product.RateIdentifier = "RUB_RUONIA_OIS"
    req_ex.Product.FundingSpread = 0.015
    req_ex.Product.FundingLegPaymentDates = list(req_ex.Product.PaymentDates)
    mds = req_ex.Model.MarketDataSet
    mds.Spots = {"ZZ_RUB": 100.0, "AA_RUB": 100.0}
    mds.Correlations = {"ZZ_RUB&AA_RUB": 0.45}
    mds.VolatilitySurfaces["ZZ_RUB"] = mds.VolatilitySurfaces.pop("EQ00_RUB")
    zz_surf = mds.VolatilitySurfaces["ZZ_RUB"]
    assert isinstance(zz_surf, EquityVolSmileCollectionSpec)
    zz_surf.Identifier = "ZZ_RUB"
    mds.VolatilitySurfaces["AA_RUB"] = mds.VolatilitySurfaces.pop("EQ01_RUB")
    aa_surf = mds.VolatilitySurfaces["AA_RUB"]
    assert isinstance(aa_surf, EquityVolSmileCollectionSpec)
    aa_surf.Identifier = "AA_RUB"
    req_ex.Product.PredefinedDividends = None

    st_ex = preprocess_eq_autocall(req_ex)
    res_gpu_ex = solver.simulate_and_price(
        st_ex,
        DupireQMCConfig(n_paths=8192, n_replicates=8, n_substeps=8, cv_mode="multifidelity_gbm"),
    )
    res_cpu_ex = price_eq_autocall_dupire_qmc(
        st_ex,
        n_paths=8192,
        n_replicates=8,
        mode="dupire_lv",
        n_substeps_per_period=8,
        seed=42,
    )
    assert abs(res_gpu_ex.product_value - res_cpu_ex.product_value) <= max(
        5e-4, 4.0 * max(res_gpu_ex.standard_error, res_cpu_ex.standard_error)
    )

    # 2. High-dimension Philox fallback (dim >= 21,201) + peak VRAM <= 250 MB delta check
    torch.cuda.synchronize()
    base_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    z_hi = _generate_gpu_sobol_tile(
        n_rep=1024,
        n_replicates=8,
        n_dims=64,
        dim_offset=21205,
        seed=42,
        device=torch.device("cuda"),
        use_triton=True,
    )
    assert z_hi.shape == (8, 64, 1024)
    assert torch.isfinite(z_hi).all()

    req_hi = _build_synthetic_request(parsed_dag, d_assets=10, m_periods=48)
    st_hi = preprocess_eq_autocall(req_hi, d_max=10, m_max=48)
    res_hi = solver.simulate_and_price(
        st_hi,
        DupireQMCConfig(n_paths=32768, n_replicates=8, n_substeps=16, cv_mode="bs1d_exact"),
    )
    assert np.isfinite(res_hi.product_value) and res_hi.standard_error > 0.0
    delta_peak_mb = (torch.cuda.max_memory_allocated() - base_bytes) / (1024.0 * 1024.0)
    assert delta_peak_mb <= 250.0, f"GPU QMC peak VRAM delta {delta_peak_mb:.1f} MB > 250.0 MB"


@pytest.mark.gpu
def test_gpu_dupire_qmc_latency_and_training_batch_teacher(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for GPU Dupire QMC verification")

    solver = DupireQMCGpuSolver(device="cuda", backend="auto")
    st_m1 = preprocess_eq_autocall(parsed_dag[0])
    st_m12 = preprocess_eq_autocall(_build_synthetic_request(parsed_dag, d_assets=5, m_periods=12, div_amt=2.0))
    st_m4 = preprocess_eq_autocall(_build_synthetic_request(parsed_dag, d_assets=3, m_periods=4))
    st_m0 = copy.copy(st_m1)
    st_m0.m_rem = 0

    cfg_32k = DupireQMCConfig(n_paths=32768, n_replicates=8, n_substeps=16, cv_mode="bs1d_exact")
    # Warmup Triton/CUDA kernels
    _ = solver.simulate_and_price(st_m1, cfg_32k)
    _ = solver.simulate_and_price(st_m12, cfg_32k)
    torch.cuda.synchronize()

    if sys.gettrace() is None:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        _ = solver.simulate_and_price(st_m1, cfg_32k)
        torch.cuda.synchronize()
        ms_m1 = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        _ = solver.simulate_and_price(st_m12, cfg_32k)
        torch.cuda.synchronize()
        ms_m12 = (time.perf_counter() - t1) * 1000.0
        vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        assert ms_m1 <= 15.0, f"GPU QMC M=1 latency {ms_m1:.2f} ms > 15.0 ms"
        assert ms_m12 < 35.0, f"GPU QMC M=12 latency {ms_m12:.2f} ms >= 35.0 ms"
        assert vram_mb <= 250.0, f"GPU QMC peak VRAM {vram_mb:.1f} MB > 250.0 MB"

    # Training batch teacher across heterogeneous (D, M_rem) buckets
    states: list[AutocallPreprocState] = [st_m1, st_m12, st_m4, st_m0] * 4
    pv, p_mc, q_mc, l_mc, delta_mc = solver.simulate_batch_for_training(
        states,
        config=DupireQMCConfig(n_paths=4096, n_replicates=8, n_substeps=8, cv_mode="none"),
        compute_pathwise_delta=True,
    )
    assert pv.dtype == torch.float64 and pv.shape == (16,)
    assert p_mc.shape == (16, 48) and q_mc.shape == (16, 48) and l_mc.shape == (16,)
    assert delta_mc is not None and delta_mc.shape == (16, 8)
    assert torch.isfinite(pv).all() and torch.isfinite(delta_mc).all()
