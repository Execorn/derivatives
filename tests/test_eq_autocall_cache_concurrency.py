"""Concurrency, thread-safety, cache immutability, and lock verification suite."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from deepvol.api.schemas_eq_autocall import EQAutocallableTradeRequest, parse_murex_autocall_dag
from deepvol.models.autocall import (
    compute_analytical_autocall_baseline,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.pre_post_processor import ThreadSafeLRUCache

_LOCAL_DATA_REQUEST = Path(__file__).resolve().parents[1] / "data" / "autocall" / "eq_autocallable_request.txt"
_MAIN_REPO_REQUEST = Path("/home/execorn/programming/derivatives/data/autocall/eq_autocallable_request.txt")
REQUEST_TXT_PATH = _LOCAL_DATA_REQUEST if _LOCAL_DATA_REQUEST.exists() else _MAIN_REPO_REQUEST


@pytest.fixture(scope="module")
def base_request() -> EQAutocallableTradeRequest:
    req, _ = parse_murex_autocall_dag(REQUEST_TXT_PATH)
    return req


def test_thread_safe_lru_cache_16_thread_stress_and_copy_isolation() -> None:
    cache = ThreadSafeLRUCache(maxsize=16, copy_fn=lambda arr: arr.copy())

    def _worker(tid: int) -> bool:
        for i in range(64):
            key = f"k_{(tid + i) % 24}"
            val = np.full(4, float(tid * 100 + i), dtype=np.float64)
            out = cache.put(key, val)
            out.fill(-999.0)
            fetched = cache.get(key)
            if fetched is not None:
                if float(fetched[0]) == -999.0:
                    return False
                fetched.fill(-888.0)
            _ = key in cache
            if i % 17 == 0:
                try:
                    _ = cache[key]
                except KeyError:
                    pass
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_worker, range(16)))
    assert all(results)
    assert len(cache) <= 16
    cache.clear()
    assert len(cache) == 0


def test_preprocess_eq_autocall_16_thread_concurrency_and_immutability(
    base_request: EQAutocallableTradeRequest,
) -> None:
    variants: list[EQAutocallableTradeRequest] = []
    expected_pvs: list[float] = []
    expected_spots: list[float] = []
    for k in range(8):
        req_k = copy.deepcopy(base_request)
        first_uid = req_k.Product.UnderlyingIdentifiers[0]
        req_k.Model.MarketDataSet.Spots[first_uid] = float(req_k.Model.MarketDataSet.Spots[first_uid]) * (
            0.90 + 0.03 * k
        )
        st_ref = preprocess_eq_autocall(req_k)
        variants.append(req_k)
        expected_pvs.append(float(st_ref.baseline.product_value_proxy.item()))
        expected_spots.append(float(st_ref.spots_t0[0].item()))

    def _run_worker(tid: int) -> None:
        for step in range(4):
            idx = (tid + step) % len(variants)
            st = preprocess_eq_autocall(variants[idx])
            assert abs(float(st.baseline.product_value_proxy.item()) - expected_pvs[idx]) < 1e-12
            assert abs(float(st.spots_t0[0].item()) - expected_spots[idx]) < 1e-12
            for surf in st.vol_surfaces.values():
                assert surf.quotes.flags.writeable is False
                assert surf.strikes.flags.writeable is False
                assert surf.tenors.flags.writeable is False
            assert st.discount_curve.tenors.flags.writeable is False
            assert st.discount_curve.rates.flags.writeable is False
            # Mutating returned mutable tensors must not poison cached artifacts for other threads
            st.spots_t0.fill_(1.0)
            st.node_feats.zero_()

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(_run_worker, tid) for tid in range(16)]
        for fut in futures:
            fut.result()

    # Post-stress integrity re-check
    for idx, req_k in enumerate(variants):
        st_after = preprocess_eq_autocall(req_k)
        assert float(st_after.baseline.product_value_proxy.item()) == pytest.approx(expected_pvs[idx], abs=1e-12)
        assert float(st_after.spots_t0[0].item()) == pytest.approx(expected_spots[idx], abs=1e-12)


def test_analytical_baseline_16_thread_no_buffer_aliasing(
    base_request: EQAutocallableTradeRequest,
) -> None:
    st_list = []
    ref_outputs = []
    for k in range(4):
        req_k = copy.deepcopy(base_request)
        for uid in req_k.Product.UnderlyingIdentifiers:
            req_k.Model.MarketDataSet.Spots[uid] = float(req_k.Model.MarketDataSet.Spots[uid]) * (0.88 + 0.08 * k)
        st_k = preprocess_eq_autocall(req_k)
        st_list.append(st_k)
        ref_outputs.append(float(compute_analytical_autocall_baseline(st_k, use_fast_cpu=True).product_value_proxy.item()))

    def _run_baseline(tid: int) -> None:
        for j in range(12):
            idx = (tid + j) % len(st_list)
            out = compute_analytical_autocall_baseline(st_list[idx], use_fast_cpu=True)
            assert float(out.product_value_proxy.item()) == pytest.approx(ref_outputs[idx], abs=1e-12)
            assert torch.isfinite(out.p_proxy).all()
            assert torch.isfinite(out.q_proxy).all()

    with ThreadPoolExecutor(max_workers=16) as pool:
        for fut in [pool.submit(_run_baseline, t) for t in range(16)]:
            fut.result()


def test_compiled_runner_output_clone_and_concurrency(
    base_request: EQAutocallableTradeRequest,
) -> None:
    from deepvol.surrogates.compiled_runner import CompiledSurrogateRunner
    from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

    st = preprocess_eq_autocall(base_request)
    model = MaskedEQAutocallSurrogate(d_node=16, d_global=16, d_model=64, n_egno_layers=2, n_sched_layers=1)
    model.eval()
    runner = CompiledSurrogateRunner(model, enable_cudagraphs=False, enable_inductor=False)
    t2 = st.to_tier2_float32_tensors()
    args = (
        t2["node_feats"],
        t2["edge_feats"],
        t2["asset_mask"],
        t2["sched_feats"],
        t2["sched_mask"],
        t2["global_feats"],
    )

    ref = runner.predict_residuals(*args)
    ref_p = ref[0].clone()
    # Mutating caller copy must not alter subsequent runner outputs
    ref[0].fill_(99.0)
    ref2 = runner.predict_residuals(*args)
    torch.testing.assert_close(ref2[0], ref_p, atol=1e-6, rtol=1e-6)

    def _run_pred(_: int) -> None:
        out = runner.predict_residuals(*args)
        torch.testing.assert_close(out[0], ref_p, atol=1e-6, rtol=1e-6)
        out[0].zero_()

    with ThreadPoolExecutor(max_workers=16) as pool:
        for fut in [pool.submit(_run_pred, i) for i in range(16)]:
            fut.result()

    runner.invalidate_graphs()
    out_after = runner.predict_residuals(*args)
    torch.testing.assert_close(out_after[0], ref_p, atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
def test_compiled_runner_cuda_graphs_clone_and_concurrency(
    base_request: EQAutocallableTradeRequest,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from deepvol.surrogates.compiled_runner import CompiledSurrogateRunner
    from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

    dev = torch.device("cuda")
    st = preprocess_eq_autocall(base_request)
    model = MaskedEQAutocallSurrogate(
        d_node=16, d_global=16, d_model=64, n_egno_layers=2, n_sched_layers=1
    ).to(dev)
    model.eval()
    model.freeze_eval_buffers()
    runner = CompiledSurrogateRunner(model, enable_cudagraphs=True, enable_inductor=False)

    t2 = {k: v.to(dev) for k, v in st.to_tier2_float32_tensors().items()}
    args = (
        t2["node_feats"],
        t2["edge_feats"],
        t2["asset_mask"],
        t2["sched_feats"],
        t2["sched_mask"],
        t2["global_feats"],
    )

    with torch.no_grad():
        eager = model.forward_residuals(*args)
        out1 = runner.predict_residuals(*args)
        torch.testing.assert_close(out1[0], eager[0], atol=1e-4, rtol=1e-4)
        ref_dh = out1[0].clone()
        out1[0].fill_(77.0)
        out2 = runner.predict_residuals(*args)
        torch.testing.assert_close(out2[0], ref_dh, atol=1e-6, rtol=1e-6)

    def _cuda_worker(_: int) -> None:
        with torch.no_grad():
            out_w = runner.predict_residuals(*args)
            torch.testing.assert_close(out_w[0], ref_dh, atol=1e-6, rtol=1e-6)
            out_w[0].zero_()

    with ThreadPoolExecutor(max_workers=16) as pool:
        for fut in [pool.submit(_cuda_worker, i) for i in range(16)]:
            fut.result()

    # Exercise grad-enabled fallback and out-of-bucket geometry fallback
    with torch.enable_grad():  # type: ignore[no-untyped-call]
        node_grad = args[0].detach().clone().requires_grad_(True)
        out_grad = runner.predict_residuals(node_grad, *args[1:])
        assert out_grad[0].requires_grad

    big_args = (
        torch.randn(1, 20, 16, device=dev, dtype=torch.float32),
        torch.randn(1, 20, 20, 4, device=dev, dtype=torch.float32),
        torch.ones(1, 20, device=dev, dtype=torch.float32),
        torch.randn(1, 60, 12, device=dev, dtype=torch.float32),
        torch.ones(1, 60, device=dev, dtype=torch.float32),
        torch.randn(1, 16, device=dev, dtype=torch.float32),
    )
    with torch.no_grad():
        out_big = runner.predict_residuals(*big_args)
        assert out_big[0].shape == (1, 60)

        runner.invalidate_graphs()
        out3 = runner.predict_residuals(*args)
        torch.testing.assert_close(out3[0], ref_dh, atol=1e-6, rtol=1e-6)


def test_async_rwlock_cross_loop_and_cancellation_wakeup() -> None:
    from deepvol.api.server import AsyncReadWriteLock

    async def _scenario() -> None:
        lock = AsyncReadWriteLock()
        events: list[str] = []

        async with lock.reader():
            events.append("r1_in")
            w_task = asyncio.create_task(_acquire_write(lock, events))
            await asyncio.sleep(0.02)
            r_queued = asyncio.create_task(_acquire_read(lock, events))
            await asyncio.sleep(0.02)
            w_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await w_task
        await r_queued
        assert "w_acquired" not in events
        assert "r2_acquired" in events

    async def _acquire_write(lock: Any, events: list[str]) -> None:
        async with lock.writer():
            events.append("w_acquired")

    async def _acquire_read(lock: Any, events: list[str]) -> None:
        async with lock.reader():
            events.append("r2_acquired")

    asyncio.run(_scenario())
    # Reusing the same AsyncReadWriteLock across a brand new event loop must succeed
    asyncio.run(_scenario())


def test_greeks_engine_16_thread_market_diversity_and_cache_eviction(
    base_request: EQAutocallableTradeRequest,
) -> None:
    from deepvol.models.autocall.greeks_engine import compute_eq_autocall_greeks
    from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

    surr = MaskedEQAutocallSurrogate(
        d_max=8, m_max=48, d_model=32, n_egno_layers=1, n_sched_layers=1
    ).eval()
    surr.zero_init_residual_heads(std=1e-3)

    requests: list[EQAutocallableTradeRequest] = []
    ref_pvs: list[float] = []
    tkr0 = base_request.Product.UnderlyingIdentifiers[0]
    for i in range(24):
        r_i = copy.deepcopy(base_request)
        r_i.Model.MarketDataSet.Spots[tkr0] *= 0.80 + 0.02 * i
        cid = next(iter(r_i.Model.MarketDataSet.RatesCurvesBundles))
        r_i.Model.MarketDataSet.RatesCurvesBundles[cid].RatesSchedule.Rates[0] += 0.05 * i
        requests.append(r_i)
        res_ref = compute_eq_autocall_greeks(
            r_i, queries=["ProductValue", "Delta", "Rho"], surrogate=surr
        )
        ref_pvs.append(float(res_ref.product_value))

    def _thread_worker(idx: int) -> None:
        for k in range(8):
            m_idx = (idx + k * 3) % len(requests)
            res = compute_eq_autocall_greeks(
                requests[m_idx], queries=["ProductValue", "Delta", "Rho"], surrogate=surr
            )
            assert abs(float(res.product_value) - ref_pvs[m_idx]) < 1e-10
            assert res.delta_tensor is not None and torch.isfinite(res.delta_tensor).all()

    with ThreadPoolExecutor(max_workers=16) as pool:
        for fut in [pool.submit(_thread_worker, t_id) for t_id in range(16)]:
            fut.result()

    # Cache eviction thrashing across > 512 distinct surface/curve inputs preserves exact results.
    state_0 = preprocess_eq_autocall(requests[0])
    pv_0_before = float(compute_eq_autocall_greeks(state_0, queries=["ProductValue"]).product_value)
    for j in range(520):
        r_j = copy.deepcopy(base_request)
        r_j.Model.MarketDataSet.Spots[tkr0] *= 0.70 + 1e-4 * j
        _ = preprocess_eq_autocall(r_j).state_digest
    pv_0_after = float(compute_eq_autocall_greeks(state_0, queries=["ProductValue"]).product_value)
    assert pv_0_after == pytest.approx(pv_0_before, abs=1e-12)


def test_arbitrage_cache_isolation_strict_vs_repaired(
    base_request: EQAutocallableTradeRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies `_SURFACE_ARB_FAST_CACHE` isolates strict rejections from `repair_arbitrage=True` clones under concurrency."""
    from deepvol.api.eq_autocall_runtime import EQAutocallRuntime
    import deepvol.models.autocall.dupire_qmc_solver as qmc_solver_mod

    req_bad = copy.deepcopy(base_request)
    tkr0 = req_bad.Product.UnderlyingIdentifiers[0]
    surf0 = req_bad.Model.MarketDataSet.VolatilitySurfaces[tkr0]
    q_bad = copy.deepcopy(surf0.VolatilityQuotes)
    mid_j = len(surf0.Strikes) // 2
    for row in q_bad:
        row[mid_j] = min(float(row[mid_j]) + 0.35, 4.50)
    surf0.VolatilityQuotes = q_bad

    def _mock_repair(strikes: Any, quotes: Any, t_exp: Any, k_ref: float, **kw: Any) -> tuple[Any, dict[str, Any]]:
        _ = (strikes, t_exp, k_ref, kw)
        return np.full_like(quotes, 0.25, dtype=np.float64), {"repaired": True}

    monkeypatch.setattr(qmc_solver_mod, "repair_arbitrage_quotes", _mock_repair, raising=False)
    rt = EQAutocallRuntime(device="cpu")

    def _check_worker(idx: int) -> float:
        if idx % 2 == 0:
            with pytest.raises(ValueError, match="SMILE_ARBITRAGE"):
                rt.price_trade(req_bad, engine="proxy", include_all_greeks=False, repair_arbitrage=False)
            return -1.0
        res = rt.price_trade(req_bad, engine="proxy", include_all_greeks=False, repair_arbitrage=True)
        return float(res.ProductValue)

    with ThreadPoolExecutor(max_workers=8) as pool:
        vals = list(pool.map(_check_worker, range(16)))
    assert all(vals[i] == -1.0 for i in range(0, 16, 2))
    assert all(np.isfinite(vals[i]) and vals[i] > 0.0 for i in range(1, 16, 2))


@pytest.mark.gpu
def test_cuda_packed_surrogate_overlapping_forwards_and_tf32_restoration(
    base_request: EQAutocallableTradeRequest,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for overlapping GPU surrogate & TF32 tests")

    from deepvol.models.autocall.gpu_cache_manager import GLOBAL_GPU_CACHE_MANAGER
    from deepvol.models.autocall.greeks_engine import evaluate_differentiable_pv
    from deepvol.surrogates.eq_autocall_egno import MaskedEQAutocallSurrogate

    dev = torch.device("cuda")
    surr = MaskedEQAutocallSurrogate(
        d_max=8, m_max=48, d_model=64, n_egno_layers=2, n_sched_layers=1
    ).to(device=dev).eval()
    surr.zero_init_residual_heads(std=1e-2)

    prev_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    prev_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with GLOBAL_GPU_CACHE_MANAGER.tf32_precision_guard(False):
            assert bool(torch.backends.cuda.matmul.allow_tf32) is False
        assert bool(torch.backends.cuda.matmul.allow_tf32) is True
        assert bool(torch.backends.cudnn.allow_tf32) is True

        # Overlapping forwards with distinct basket dimensions and masks must not corrupt backward graphs.
        req_small = copy.deepcopy(base_request)
        tks2 = req_small.Product.UnderlyingIdentifiers[:2]
        req_small.Product.UnderlyingIdentifiers = tks2
        req_small.Product.UnderlyingReferenceSpots = req_small.Product.UnderlyingReferenceSpots[:2]
        req_small.Model.MarketDataSet.Spots = {k: req_small.Model.MarketDataSet.Spots[k] for k in tks2}
        req_small.Model.MarketDataSet.VolatilitySurfaces = {
            k: req_small.Model.MarketDataSet.VolatilitySurfaces[k] for k in tks2
        }
        req_small.Model.MarketDataSet.Dividends = {
            k: req_small.Model.MarketDataSet.Dividends[k]
            for k in tks2
            if k in req_small.Model.MarketDataSet.Dividends
        }
        req_small.Model.MarketDataSet.Fixings = {
            k: req_small.Model.MarketDataSet.Fixings[k]
            for k in tks2
            if k in req_small.Model.MarketDataSet.Fixings
        }
        req_small.Model.MarketDataSet.Correlations = {f"{tks2[0]}&{tks2[1]}": 0.4}

        st1 = preprocess_eq_autocall(base_request)
        st2 = preprocess_eq_autocall(req_small)
        sp1 = st1.spots_t0.detach().to(device=dev).requires_grad_(True)
        sp2 = st2.spots_t0.detach().to(device=dev).requires_grad_(True)
        pv1 = evaluate_differentiable_pv(st1, spots_t0=sp1, surrogate=surr)
        pv2 = evaluate_differentiable_pv(st2, spots_t0=sp2, surrogate=surr)
        assert isinstance(pv1, torch.Tensor) and isinstance(pv2, torch.Tensor)
        g1, g2 = torch.autograd.grad(pv1 + pv2, [sp1, sp2])

        sp1_solo = st1.spots_t0.detach().to(device=dev).requires_grad_(True)
        pv1_solo = evaluate_differentiable_pv(st1, spots_t0=sp1_solo, surrogate=surr)
        assert isinstance(pv1_solo, torch.Tensor)
        g1_solo = torch.autograd.grad(pv1_solo, sp1_solo)[0]
        torch.testing.assert_close(g1, g1_solo, atol=1e-5, rtol=1e-5)
        assert torch.isfinite(g2).all()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

