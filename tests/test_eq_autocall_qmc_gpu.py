from __future__ import annotations

import copy
import datetime
import gc
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
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
        "EquallyWeightedBasket",
        "SecondWorstOf",
        "ThirdWorstOf",
        "SecondBestOf",
        "UnknownCustom",
    )
    for idx_name in all_idx:
        w3 = _evaluate_monitoring_index_gpu(perf3, idx_name, perm3, asset_dim=-1)
        w3g, g3 = _eval_index_and_grad_gpu(perf3, idx_name, perm3, compute_grad=True, asset_dim=-1)
        w3n, g3n = _eval_index_and_grad_gpu(perf3, idx_name, perm3, compute_grad=False, asset_dim=-1)
        torch.testing.assert_close(w3, w3g, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w3, w3n, atol=1e-5, rtol=1e-5)
        assert g3 is not None and torch.isfinite(g3).all() and g3n is None
        w1_val = _evaluate_monitoring_index_gpu(perf1, idx_name, None, asset_dim=-1)
        _, g1 = _eval_index_and_grad_gpu(perf1, idx_name, None, compute_grad=True, asset_dim=-1)
        assert g1 is not None and g1.shape == (1, 1)
        if idx_name == "SecondWorstOf":
            assert float(w3[0]) == pytest.approx(1.1, abs=1e-12)
        elif idx_name == "ThirdWorstOf":
            assert float(w3[0]) == pytest.approx(1.3, abs=1e-12)
        elif idx_name == "SecondBestOf":
            assert float(w3[0]) == pytest.approx(1.1, abs=1e-12)
        elif idx_name == "EquallyWeightedBasket":
            assert float(w3[0]) == pytest.approx(1.1, abs=1e-12)
        elif idx_name in ("DiffAB", "DiffBA", "Dispersion"):
            assert float(w1_val[0]) == pytest.approx(0.0, abs=1e-12)
        elif idx_name in ("RatioAB", "RatioBA"):
            assert float(w1_val[0]) == pytest.approx(1.05, abs=1e-12)

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


def test_gpu_cache_manager_contract_and_cpu_parity_blind_spots(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    from dataclasses import dataclass
    from deepvol.models.autocall.gpu_cache_manager import (
        GLOBAL_GPU_CACHE_MANAGER,
        PartitionedGpuLRU,
        measure_tensor_nbytes,
        normalize_cuda_device,
    )

    @dataclass
    class _Bundle:
        a: torch.Tensor
        b: tuple[torch.Tensor, ...]

    t1 = torch.zeros(16, dtype=torch.float64)
    t2 = torch.zeros(8, dtype=torch.float32)
    bundle = _Bundle(a=t1, b=(t2,))
    assert measure_tensor_nbytes(bundle) == 16 * 8 + 8 * 4

    evicted: list[Any] = []
    lru: PartitionedGpuLRU[torch.Tensor] = PartitionedGpuLRU(
        "unit_test", max_entries=4, max_bytes=256, on_evict=lambda k, _v: evicted.append(k)
    )
    lru.put("ok1", torch.zeros(16, dtype=torch.float64))
    assert lru.resident_bytes == 128 and len(lru) == 1
    lru.put("too_big", torch.zeros(64, dtype=torch.float64))
    assert lru.resident_bytes <= 256 and "too_big" not in lru._store
    assert GLOBAL_GPU_CACHE_MANAGER.TOTAL_BUDGET_BYTES <= 192 * 1024 * 1024
    assert normalize_cuda_device("cpu") == ("cpu", -1)

    # Already-autocalled state must yield finite zeroed training deltas rather than pricing dead periods.
    req_ko = copy.deepcopy(parsed_dag[0])
    for tkr, ref_s in zip(req_ko.Product.UnderlyingIdentifiers, req_ko.Product.UnderlyingReferenceSpots):
        req_ko.Model.MarketDataSet.Fixings[tkr].Values[-1] = float(ref_s) * 1.5
    st_ko = preprocess_eq_autocall(req_ko)
    cpu_solver = CpuNumpyDupireQMC()
    pv_tr, _, _, _, delta_tr = cpu_solver.simulate_batch_for_training(
        [st_ko],
        config=DupireQMCConfig(n_paths=256, n_replicates=2, n_substeps=2),
        compute_pathwise_delta=True,
    )
    assert torch.isfinite(pv_tr).all() and delta_tr is not None and torch.all(delta_tr == 0.0)


@pytest.mark.gpu
def test_gpu_qmc_vram_m60_d12_qshift_onestar_and_nonpow2_replicates(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")

    from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER

    solver = DupireQMCGpuSolver(device="cuda", backend="triton")

    # Non-pow2 n_replicates=3 (BLOCK_N alignment) & multifidelity_gbm on M=60, d=12 <= 250 MB VRAM.
    req_m60_d12 = _build_synthetic_request(parsed_dag, d_assets=12, m_periods=60, div_amt=1.5)
    st_m60_d12 = preprocess_eq_autocall(req_m60_d12, d_max=12, m_max=60)
    GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_res = torch.cuda.memory_reserved()

    res_np2 = solver.simulate_and_price(
        st_m60_d12,
        DupireQMCConfig(n_paths=1536, n_replicates=3, n_substeps=4, cv_mode="bs1d_exact"),
    )
    assert np.isfinite(res_np2.product_value)

    res_mf = solver.simulate_and_price(
        st_m60_d12,
        DupireQMCConfig(n_paths=8192, n_replicates=8, n_substeps=4, cv_mode="multifidelity_gbm"),
    )
    torch.cuda.synchronize()
    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    max_res = torch.cuda.max_memory_reserved()
    assert np.isfinite(res_mf.product_value)
    assert peak_alloc_mb <= 237.0, f"M=60 d=12 peak allocated {peak_alloc_mb:.1f} MB > 237 MB"
    if base_res > 64 * 1024 * 1024 and os.environ.get("_DEEPVOL_VRAM_SUBPROC") != "1":
        delta_res_mb = (max_res - base_res) / (1024.0 * 1024.0)
        assert delta_res_mb <= 250.0, (
            f"M=60 d=12 delta reserved {delta_res_mb:.1f} MB > 250 MB (base={base_res / (1024.0 * 1024.0):.1f} MB)"
        )
        sub_env = os.environ.copy()
        sub_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        sub_env["_DEEPVOL_VRAM_SUBPROC"] = "1"
        sub_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), str(REPO_ROOT / "src")]
            + ([sub_env["PYTHONPATH"]] if sub_env.get("PYTHONPATH") else [])
        )
        sub_code = textwrap.dedent(
            f"""
            import gc, sys, torch
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            from deepvol.api.schemas_eq_autocall import parse_murex_autocall_dag
            from deepvol.models.autocall import preprocess_eq_autocall
            from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig, DupireQMCGpuSolver
            from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER
            from test_eq_autocall_qmc_gpu import REF_REQUEST_PATH, _build_synthetic_request
            solver = DupireQMCGpuSolver(device="cuda", backend="triton")
            req_m60_d12 = _build_synthetic_request(parse_murex_autocall_dag(REF_REQUEST_PATH), d_assets=12, m_periods=60, div_amt=1.5)
            st_m60_d12 = preprocess_eq_autocall(req_m60_d12, d_max=12, m_max=60)
            GLOBAL_GPU_CACHE_MANAGER.clear_all(clear_canonical_models=True)
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            solver.simulate_and_price(st_m60_d12, DupireQMCConfig(n_paths=1536, n_replicates=3, n_substeps=4, cv_mode="bs1d_exact"))
            solver.simulate_and_price(st_m60_d12, DupireQMCConfig(n_paths=8192, n_replicates=8, n_substeps=4, cv_mode="multifidelity_gbm"))
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated() <= 237.0 * 1024 * 1024
            assert torch.cuda.max_memory_reserved() <= 250.0 * 1024 * 1024
            assert GLOBAL_GPU_CACHE_MANAGER.total_resident_bytes() <= GLOBAL_GPU_CACHE_MANAGER.TOTAL_BUDGET_BYTES
            """
        )
        subprocess.run([sys.executable, "-c", sub_code], env=sub_env, check=True, timeout=120)
    else:
        peak_res_mb = max_res / (1024.0 * 1024.0)
        assert peak_res_mb <= 250.0, f"M=60 d=12 peak reserved {peak_res_mb:.1f} MB > 250 MB"
    assert GLOBAL_GPU_CACHE_MANAGER.total_resident_bytes() <= GLOBAL_GPU_CACHE_MANAGER.TOTAL_BUDGET_BYTES

    # Quanto training q_shift != 0 & OneStar pathwise Delta finite-difference parity on GPU.
    req_q_os = copy.deepcopy(parsed_dag[0])
    req_q_os.Product.PaymentCurrency = "USD"
    req_q_os.Product.OneStarStrike = 1.02
    req_q_os.Product.OneStarCoupon = 0.05
    st_q_os = preprocess_eq_autocall(req_q_os)
    cfg_tr = DupireQMCConfig(n_paths=2048, n_replicates=4, n_substeps=4, cv_mode="none")
    pv_base, _, _, _, delta_base = solver.simulate_batch_for_training(
        [st_q_os], config=cfg_tr, compute_pathwise_delta=True
    )
    assert torch.isfinite(pv_base).all() and delta_base is not None and torch.isfinite(delta_base).all()
    h_rel = 1e-3
    req_up = copy.deepcopy(req_q_os)
    first_tkr = req_up.Product.UnderlyingIdentifiers[0]
    req_up.Model.MarketDataSet.Spots[first_tkr] *= 1.0 + h_rel
    req_dn = copy.deepcopy(req_q_os)
    req_dn.Model.MarketDataSet.Spots[first_tkr] *= 1.0 - h_rel
    pv_up, _, _, _, _ = solver.simulate_batch_for_training(
        [preprocess_eq_autocall(req_up)], config=cfg_tr, compute_pathwise_delta=False
    )
    pv_dn, _, _, _, _ = solver.simulate_batch_for_training(
        [preprocess_eq_autocall(req_dn)], config=cfg_tr, compute_pathwise_delta=False
    )
    idx_sorted = st_q_os.sorted_identifiers.index(first_tkr)
    fd_log_delta = (float(pv_up[0]) - float(pv_dn[0])) / (2.0 * h_rel)
    assert abs(float(delta_base[0, idx_sorted]) - fd_log_delta) < 0.05


def test_global_gpu_cache_manager_112mb_6_partitions_and_eviction() -> None:
    """Enforces 112 MB total budget, 6 named partitions, oversized guard, and <=8-entry cudagraph LRU."""
    from deepvol.models.autocall.gpu_cache_manager import (
        GLOBAL_GPU_CACHE_MANAGER,
        GlobalGpuCacheManager,
    )

    mgr = GlobalGpuCacheManager()
    assert mgr.TOTAL_BUDGET_BYTES == 112 * 1024 * 1024
    expected_budgets = {
        "static_constants": 8 * 1024 * 1024,
        "cudagraph_pool": 24 * 1024 * 1024,
        "triton_kernels": 12 * 1024 * 1024,
        "sobol_normals": 20 * 1024 * 1024,
        "deal_prep_lv_vega": 32 * 1024 * 1024,
        "lv_table_cache": 16 * 1024 * 1024,
    }
    assert sum(expected_budgets.values()) == 112 * 1024 * 1024
    for name, b in expected_budgets.items():
        part = mgr.register_partition(name, max_entries=64, max_bytes=b * 2)
        assert part.max_bytes == b
    assert GLOBAL_GPU_CACHE_MANAGER.TOTAL_BUDGET_BYTES == 112 * 1024 * 1024

    cg_pool = mgr.register_cudagraph_pool(max_entries=8)
    assert cg_pool.max_entries == 8
    assert cg_pool.max_bytes == 24 * 1024 * 1024
    evicted_cg: list[str] = []
    cg_pool_test = mgr.register_partition(
        "cudagraph_pool_test", max_entries=8, max_bytes=1024, on_evict=lambda k, _v: evicted_cg.append(str(k))
    )
    for i in range(10):
        cg_pool_test.put(f"shape_{i}", torch.zeros(16, dtype=torch.float32))
    assert len(cg_pool_test) <= 8
    assert evicted_cg == ["shape_0", "shape_1"]

    # Oversized tensor must be rejected safely without corrupting LRU accounting.
    cg_pool_test.put("ok_small", torch.zeros(8, dtype=torch.float32))
    before_bytes = cg_pool_test.resident_bytes
    before_len = len(cg_pool_test)
    cg_pool_test.put("oversized", torch.zeros(1024, dtype=torch.float32))
    assert cg_pool_test.resident_bytes == before_bytes
    assert len(cg_pool_test) == before_len
    assert cg_pool_test.get("oversized") is None

    mf_lru = mgr.register_mf_cheap_cache(max_entries=4, max_bytes=256)
    mf_lru.put("k1", torch.zeros(8, dtype=torch.float64))
    assert mf_lru.resident_bytes == 64
    mgr.evict_partition("deal_prep_lv_vega", clear=True)
    assert mf_lru.resident_bytes == 0


def test_bf37d79_helpers_haar_bb_and_canonical_config(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """Verifies ported bf37d79 helpers (_arr_bytes_equal, _fast_triton_launch, Haar BB, CanonicalTier3Config)."""
    from types import SimpleNamespace

    from deepvol.models.autocall.dupire_qmc_gpu import (
        _HAS_TRITON,
        GpuTorchDupireQMC,
        _arr_bytes_equal,
        _fast_triton_launch,
        _haar_global_dims_for_period,
        _haar_reconstruct_residuals_torch,
        _sobol_vec_flat,
        _splitmix64_30bit_shifts,
        resolve_dupire_qmc_config,
    )

    assert callable(_fast_triton_launch)
    if _HAS_TRITON:
        assert _sobol_vec_flat is not None
    a = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    c = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    assert _arr_bytes_equal(a, b)
    assert not _arr_bytes_equal(a, c)
    assert not _arr_bytes_equal(a, np.asarray([1.0, 2.0], dtype=np.float64))

    keys = np.asarray([10, 20, 30], dtype=np.uint64)
    sh1 = _splitmix64_30bit_shifts(keys, 4)
    sh2 = _splitmix64_30bit_shifts(keys, 4)
    assert sh1.shape == (3, 4) and sh1.dtype == np.int32
    assert np.array_equal(sh1, sh2)
    assert np.all((sh1 >= 0) & (sh1 < (1 << 30)))

    # Nested Haar global dims at n_sub=4 must be strict prefix of n_sub=8.
    d4 = _haar_global_dims_for_period(m_rem=4, d_assets=2, n_sub=4, m=1).reshape(2, 3)
    d8 = _haar_global_dims_for_period(m_rem=4, d_assets=2, n_sub=8, m=1).reshape(2, 7)
    assert np.array_equal(d4, d8[:, :3])

    gen = torch.Generator().manual_seed(7)
    modes = torch.randn((32, 3, 256), generator=gen, dtype=torch.float64)
    res = _haar_reconstruct_residuals_torch(modes, n_sub=4)
    assert res.shape == (32, 4, 256)
    assert torch.allclose(res.sum(dim=1), torch.zeros((32, 256), dtype=torch.float64), atol=1e-12)

    default_cfg = DupireQMCConfig()
    assert default_cfg.adaptive_substeps is False
    assert default_cfg.lv_grid_nt == 32 and default_cfg.lv_grid_ny == 64

    canon_duck = SimpleNamespace(
        n_paths=1024,
        n_replicates=4,
        n_substeps=2,
        cv_mode="none",
        seed=123,
        lv_grid_nt=16,
        lv_grid_ny=32,
        adaptive_substeps=False,
        n_cheap_mf=2048,
    )
    conv = DupireQMCConfig.from_canonical(canon_duck)
    assert conv.n_paths == 1024 and conv.n_replicates == 4 and conv.n_substeps == 2
    assert conv.lv_grid_nt == 16 and conv.lv_grid_ny == 32 and conv.n_cheap_mf == 2048

    auto_cfg = resolve_dupire_qmc_config(2, 6, device_type="cpu", n_fallback_paths=512)
    assert auto_cfg.n_paths == 512 and auto_cfg.adaptive_substeps is False

    st = preprocess_eq_autocall(copy.deepcopy(parsed_dag[0]))
    res_auto = GpuTorchDupireQMC(device="cpu", use_triton=False).simulate_and_price(st, canon_duck)  # type: ignore[arg-type]
    assert np.isfinite(res_auto.product_value)


def test_math_parity_v3_buehler_islongbasket_onestar_and_ki_obs(
    parsed_dag: tuple[EQAutocallableTradeRequest, dict[str, Any]],
) -> None:
    """CPU unit checks for V3 memory, Buehler drift LV, IsLongBasket I_ramp, OneStar left-spread, and ki_obs_type."""
    from types import SimpleNamespace

    from deepvol.models.autocall.dupire_qmc_gpu import (
        GpuTorchDupireQMC,
        _eval_gpu_surface_local_vol_table,
        _onestar_ramp_torch,
        _resolve_ki_obs_mode,
        _resolve_onestar_effective_params,
        _resolve_r_native_obs_tensor,
    )

    # 1) ki_obs_type None/empty must strictly default to European (never Discrete even if ki_monitored_past=True).
    assert _resolve_ki_obs_mode(SimpleNamespace(ki_obs_type=None, seasoned_state=SimpleNamespace(ki_obs_type=None, ki_monitored_past=True))) == "European"
    assert _resolve_ki_obs_mode(SimpleNamespace(ki_obs_type="Discrete", seasoned_state=None)) == "Discrete"
    assert _resolve_ki_obs_mode(SimpleNamespace(ki_obs_type="Daily", seasoned_state=None)) == "Daily"
    assert _resolve_ki_obs_mode(SimpleNamespace(ki_obs_type="Continuous", seasoned_state=None)) == "Continuous"

    # 2) OneStar shifted left-spread vs narrow centered 0.02 when ki_barrier is None.
    k_eff, eps_eff = _resolve_onestar_effective_params(1.0, 0.7, -0.05)
    assert abs(k_eff - 0.975) < 1e-12 and abs(eps_eff - 0.05) < 1e-12
    assert float(_onestar_ramp_torch(torch.tensor(0.95, dtype=torch.float64), k_eff, eps_eff)) == pytest.approx(0.0, abs=1e-12)
    assert float(_onestar_ramp_torch(torch.tensor(0.975, dtype=torch.float64), k_eff, eps_eff)) == pytest.approx(0.5, abs=1e-12)
    assert float(_onestar_ramp_torch(torch.tensor(1.00, dtype=torch.float64), k_eff, eps_eff)) == pytest.approx(1.0, abs=1e-12)

    k_none, eps_none = _resolve_onestar_effective_params(1.0, None, -0.05)
    assert abs(k_none - 1.0) < 1e-12 and abs(eps_none - 0.02) < 1e-12

    # 3) Buehler effective drift must alter LV table when discrete cash dividends are present on skewed surface.
    req_div = copy.deepcopy(parsed_dag[0])
    st_ref = preprocess_eq_autocall(req_div)
    tkr0 = st_ref.sorted_identifiers[0]
    surf0 = st_ref.vol_surfaces[tkr0]
    tau_arr = st_ref.tau_obs.detach().cpu().numpy()
    r_nat = _resolve_r_native_obs_tensor(st_ref)[0].detach().cpu().numpy()
    df_fin = st_ref.dividend_schedule.df_fin_obs[0].detach().cpu().numpy()
    spot0 = float(st_ref.spots_t0[0].item())
    vt0 = float(st_ref.v_perf[0, -1].item())
    lv_nodiv, _, _ = _eval_gpu_surface_local_vol_table(
        surf0, spot0, r_nat, df_fin, tau_arr, [], vt0, 16, 32, torch.device("cpu")
    )
    synth_divs = [SimpleNamespace(tau_ex_act365=float(tau_arr[0]) * 0.5, amount=spot0 * 0.08, pv_discount_weight=0.98)]
    lv_withdiv, _, _ = _eval_gpu_surface_local_vol_table(
        surf0, spot0, r_nat, df_fin, tau_arr, synth_divs, vt0, 16, 32, torch.device("cpu")
    )
    assert float((lv_withdiv - lv_nodiv).abs().max().item()) > 1e-4

    # 4) V3 literal memory + IsLongBasket parity between GpuTorchDupireQMC(cpu) and CpuNumpyDupireQMC.
    req_mem = copy.deepcopy(parsed_dag[0])
    req_mem.Product.IsMemory = True
    st_mem = preprocess_eq_autocall(req_mem)
    cfg_cpu = DupireQMCConfig(n_paths=1024, n_replicates=4, n_substeps=2, mode="dupire_lv", cv_mode="none", lv_grid_nt=16, lv_grid_ny=32, seed=42)
    gpu_cpu = GpuTorchDupireQMC(device="cpu", use_triton=False)
    cpu_ref = CpuNumpyDupireQMC()
    res_gpu_mem = gpu_cpu.simulate_and_price(st_mem, cfg_cpu)
    res_cpu_mem = cpu_ref.simulate_and_price(st_mem, cfg_cpu)
    assert abs(res_gpu_mem.product_value - res_cpu_mem.product_value) < max(0.015, 4.0 * res_cpu_mem.standard_error)

    req_lb = copy.deepcopy(parsed_dag[0])
    req_lb.Product.IsLongBasket = True
    req_lb.Product.Strike = 1.0
    req_lb.Product.OptionDownAndInBarrierLevel = None
    req_lb.Product.OptionGearing = 1.0
    st_lb = preprocess_eq_autocall(req_lb)
    res_lb = gpu_cpu.simulate_and_price(st_lb, cfg_cpu)
    res_lb_ref = cpu_ref.simulate_and_price(st_lb, cfg_cpu)
    pv_lb_tr, _, _, _, delta_lb_tr = gpu_cpu.simulate_batch_for_training(
        [st_lb], config=cfg_cpu, compute_pathwise_delta=True
    )
    assert np.isfinite(res_lb.product_value) and torch.isfinite(pv_lb_tr).all()
    assert abs(res_lb.product_value - res_lb_ref.product_value) < max(0.015, 4.0 * res_lb_ref.standard_error)
    assert delta_lb_tr is not None and torch.isfinite(delta_lb_tr).all()
    assert abs(float(pv_lb_tr[0]) - res_lb.product_value) < max(0.02, 4.0 * res_lb.standard_error)


