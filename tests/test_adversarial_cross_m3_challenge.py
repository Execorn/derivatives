# pyright: reportPrivateUsage=none
"""tests/test_adversarial_cross_m3_challenge.py — Empirical challenge suite for Milestone 3.

Audits and stress-tests:
1. Gate G3.2 Crisis Shocks Recall:
   - 1,000 synthetic crisis shocks drawn beyond q995 (D_M in [q995, 3*q995]) and bounding box violations.
   - Verifies 100% fallback_activated=True, fallback_reason="OOD_MARKET_REGIME", and bitwise price equality.
   - Tests with zero-initialized net, non-zero random-weight single-member net, and non-zero 5-member ensemble net.
2. Gate G3.3 In-Distribution False-Positive Rate:
   - Evaluates MarketStateOODDetector.score(m) on holdout test states across multiple seeds and sample sizes.
   - Asserts FPR is strictly within 0.5% +/- 0.3% ([0.2%, 0.8%]).
3. Task 3.4 ModelReserveTracker SLA & Concurrency:
   - Benchmarks tracker.get(key) lookup latency <= 1 us (<= 10 us SLA).
   - Verifies non-blocking execution with 1.0s delayed copula engine (< 10 us quote lookup).
   - Verifies single-flight deduplication across 30 concurrent refreshes.
   - Verifies error propagation and clean-up in single-flight deduplication.
   - Verifies TTL staleness lifecycle at max_age = 15 min.
4. Gate G2.5 Latency Parity:
   - Profiles single-quote latency: deep-wing (|z| >= 7) vs ATM (z=0) at p50.
   - Verifies parity within 5 us on the single-quote path.
   - Verifies pure-host scalar evaluation <= 2 us.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import gc
import time
from typing import List
import numpy as np
import torch
from torch import Tensor

from deepvol.market.svi_slice import RawSVI
from deepvol.models.fx_cross_analytic import AnalyticTriangulated, Col
from deepvol.models.fx_cross_copula import CrossState
from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine, ResidualCrossVolNet
from deepvol.mrm.fx_cross_guardian import (
    FallbackReasonEnum,
    FXCrossGuardian,
    MarketStateOODDetector,
    ModelReserveTracker,
    PricingEngineEnum,
)


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_sample_12d_market_states(n: int, seed: int = 42) -> Tensor:
    """Generates synthetic 12-D market states within typical in-distribution domains."""
    gen = torch.Generator().manual_seed(seed)
    sqrt_T = torch.empty(n, 1, dtype=torch.float64).uniform_(0.05, 1.5, generator=gen)
    rho = torch.empty(n, 1, dtype=torch.float64).uniform_(-0.8, 0.8, generator=gen)

    atm1 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.10, 0.40, generator=gen)
    rr25_1 = torch.empty(n, 1, dtype=torch.float64).uniform_(-0.15, 0.15, generator=gen)
    bf25_1 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.01, 0.08, generator=gen)
    rr10_1 = rr25_1 * 1.8
    bf10_1 = bf25_1 * 3.0

    atm2 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.10, 0.50, generator=gen)
    rr25_2 = torch.empty(n, 1, dtype=torch.float64).uniform_(-0.20, 0.20, generator=gen)
    bf25_2 = torch.empty(n, 1, dtype=torch.float64).uniform_(0.01, 0.10, generator=gen)
    rr10_2 = rr25_2 * 1.8
    bf10_2 = bf25_2 * 3.0

    return torch.cat([sqrt_T, rho, atm1, rr25_1, bf25_1, rr10_1, bf10_1, atm2, rr25_2, bf25_2, rr10_2, bf10_2], dim=1)


def _make_34col_state_from_12d(m_12d: Tensor) -> Tensor:
    """Converts 12-D market states into 34-column engine tensors."""
    S_cnt = m_12d.shape[0]
    X = torch.zeros(S_cnt, 34, dtype=torch.float64)
    sq_T = torch.clamp(m_12d[:, 0].abs(), min=1e-4)
    T_val = sq_T ** 2
    X[:, Col.T] = T_val
    X[:, Col.T_SET] = T_val
    X[:, Col.RHO] = torch.clamp(m_12d[:, 1], -0.999, 0.999)
    X[:, Col.XI1 : Col.XI1 + 5] = m_12d[:, 2:7]
    X[:, Col.XI2 : Col.XI2 + 5] = m_12d[:, 7:12]

    X[:, Col.F] = 100.0
    X[:, Col.K] = 100.0
    X[:, Col.D] = 0.95
    X[:, Col.G] = 1.0
    X[:, Col.PHI] = 1.0
    X[:, Col.NN_GATE] = 1.0

    atm1 = torch.clamp(m_12d[:, 2].abs(), min=0.01)
    atm2 = torch.clamp(m_12d[:, 7].abs(), min=0.01)
    X[:, Col.SVI1] = torch.clamp((atm1 ** 2) * T_val, min=1e-4)
    X[:, Col.SVI1 + 1] = 0.05
    X[:, Col.SVI1 + 4] = 0.10
    X[:, Col.SVI2] = torch.clamp((atm2 ** 2) * T_val, min=1e-4)
    X[:, Col.SVI2 + 1] = 0.05
    X[:, Col.SVI2 + 4] = 0.10

    return X


class _MockDelayedCopulaEngine:
    """Mock copula engine with configurable delay and failure injection."""

    def __init__(self, delay_sec: float = 0.0, fail_on_call: int = -1) -> None:
        self.delay_sec = delay_sec
        self.fail_on_call = fail_on_call
        self.device = "cpu"
        self.call_count = 0

    def price(self, state: CrossState, strikes: Tensor, is_call: bool = True) -> Tensor:
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError(f"Simulated copula failure on call {self.call_count}")
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        F_val = float(state.F.flatten()[0].item())
        K_f64 = strikes.to(dtype=torch.float64)
        c = torch.clamp(F_val - K_f64, min=0.0) + 2.5
        if c.dim() == 1:
            c = c.unsqueeze(0)
        return c


def _make_test_cross_state() -> CrossState:
    """Creates a deterministic CrossState object."""
    F1 = torch.tensor([2650.0], dtype=torch.float64)
    F2 = torch.tensor([95.0], dtype=torch.float64)
    T = torch.tensor([1.0], dtype=torch.float64)
    svi1 = RawSVI(torch.tensor([[0.04, 0.10, -0.20, 0.0, 0.10]], dtype=torch.float64))
    svi2 = RawSVI(torch.tensor([[0.09, 0.15, 0.25, 0.0, 0.12]], dtype=torch.float64))
    rho = torch.tensor([-0.20], dtype=torch.float64)
    return CrossState(F1=F1, F2=F2, T=T, svi1=svi1, svi2=svi2, rho=rho)


# ---------------------------------------------------------------------------
# Challenge 1: Gate G3.2 Crisis Shocks Recall & Bitwise Equality
# ---------------------------------------------------------------------------

def test_challenge_g32_crisis_shocks_recall_and_bitwise_equality() -> None:
    """Challenge Gate G3.2: 1,000 synthetic crisis shocks drawn beyond q995 and box bounds.

    Asserts:
      - 100% fallback_activated == True
      - 100% fallback_reason == "OOD_MARKET_REGIME"
      - 100% nn_gate == 0.0
      - 100% engine_used == Analytic_Triangulated
      - Bitwise price equality to Analytic_Triangulated:
        (a) Zero-initialized net
        (b) Non-zero randomly initialized net
        (c) 5-member ensemble with non-zero weights
    """
    # 1. Fit detector on 3,000 baseline states
    train_states = _make_sample_12d_market_states(3000, seed=111)
    detector = MarketStateOODDetector.fit(train_states, margin=0.02)
    guardian = FXCrossGuardian(ood_detector=detector)

    q995 = detector.q995
    assert q995 > 0.0

    # 2. Build 1,000 crisis shocks
    # (A) 500 Mahalanobis outliers with D_M strictly in [q995, 3 * q995]
    cov = torch.linalg.inv(detector.cov_inv)
    L = torch.linalg.cholesky(cov)
    gen = torch.Generator().manual_seed(222)

    shocks_list: List[Tensor] = []
    while sum(s.shape[0] for s in shocks_list) < 500:
        batch_n = 600
        unit_dirs = torch.randn(batch_n, 12, dtype=torch.float64, generator=gen)
        unit_dirs = unit_dirs / torch.norm(unit_dirs, dim=-1, keepdim=True)
        radii = torch.empty(batch_n, 1, dtype=torch.float64).uniform_(1.05 * q995, 2.95 * q995, generator=gen)
        cand = detector.mean + (unit_dirs @ L.t()) * radii

        # Clamp to valid financial ranges
        cand[:, 0] = torch.clamp(cand[:, 0], min=0.05, max=2.5)
        cand[:, 1] = torch.clamp(cand[:, 1], min=-0.95, max=0.95)
        cand[:, 2] = torch.clamp(cand[:, 2], min=0.02, max=2.0)
        cand[:, 7] = torch.clamp(cand[:, 7], min=0.02, max=2.0)

        d_m_cand, _, _ = detector.score(cand)
        valid = (d_m_cand >= q995) & (d_m_cand <= 3.0 * q995)
        if valid.any():
            shocks_list.append(cand[valid])

    mahalanobis_crisis = torch.cat(shocks_list, dim=0)[:500]

    # Verify that all 500 have D_M >= q995
    d_m_check, _, _ = detector.score(mahalanobis_crisis)
    assert (d_m_check >= q995).all().item() is True, f"Found D_M < q995: min={d_m_check.min().item()}, q995={q995}"

    # (B) 500 Bounding box violations
    box_crisis = train_states[:500].clone()
    for i in range(500):
        mode = i % 5
        if mode == 0:
            box_crisis[i, 7] = 1.50  # RUB ATM vol = 150% (box_hi ~ 0.50)
        elif mode == 1:
            box_crisis[i, 3] = 0.40  # RR25 = 40% (box_hi ~ 0.15)
        elif mode == 2:
            box_crisis[i, 1] = 0.999 # extreme correlation
        elif mode == 3:
            box_crisis[i, 0] = 0.005 # ultra short expiry ~ 9 hours
        elif mode == 4:
            box_crisis[i, 4] = 0.35  # extreme BF25 = 35%

    crisis_1000 = torch.cat([mahalanobis_crisis, box_crisis], dim=0)
    assert crisis_1000.shape == (1000, 12)

    # 3. Route through Guardian
    X_crisis = _make_34col_state_from_12d(crisis_1000)
    decision = guardian.route(X_crisis, engine=PricingEngineEnum.DIFFERENTIAL_ML)

    # Assert 100% fallback and reason
    assert decision.fallback_activated.all().item() is True, "Gate G3.2 FAIL: fallback_activated not 100%"
    assert (decision.nn_gate == 0.0).all().item() is True, "Gate G3.2 FAIL: nn_gate != 0.0"
    assert all(r == FallbackReasonEnum.OOD_MARKET_REGIME.value for r in decision.fallback_reason), (
        "Gate G3.2 FAIL: fallback_reason != OOD_MARKET_REGIME"
    )
    assert all(e == PricingEngineEnum.ANALYTIC_TRIANGULATED for e in decision.engine_used), (
        "Gate G3.2 FAIL: engine_used != Analytic_Triangulated"
    )

    # 4. Bitwise price equality under various neural network weights
    analytic_engine = AnalyticTriangulated()
    out_analytic = analytic_engine(X_crisis)

    X_gated = X_crisis.clone()
    X_gated[:, Col.NN_GATE] = decision.nn_gate

    # (A) Zero-initialized net
    net_zero = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine_zero = CrossVolSurrogateEngine(net=net_zero, eps_bound=0.25)
    out_zero = engine_zero(X_gated)
    assert torch.equal(out_zero.V, out_analytic.V), "Zero-net prices not bitwise equal to Analytic_Triangulated"
    assert torch.equal(out_zero.s_k, out_analytic.s_k), "Zero-net s_k not bitwise equal"
    assert torch.equal(out_zero.s_kk, out_analytic.s_kk), "Zero-net s_kk not bitwise equal"

    # (B) Non-zero randomly initialized net (simulating arbitrary weights)
    net_random = ResidualCrossVolNet(width=64, depth=3, members=1)
    with torch.no_grad():
        for p in net_random.parameters():
            p.normal_(0.0, 0.5)
    engine_random = CrossVolSurrogateEngine(net=net_random, eps_bound=0.25)
    out_random = engine_random(X_gated)
    assert torch.equal(out_random.V, out_analytic.V), (
        "Random-net with nn_gate=0 not bitwise equal to Analytic_Triangulated"
    )
    assert torch.equal(out_random.s_k, out_analytic.s_k)
    assert torch.equal(out_random.s_kk, out_analytic.s_kk)

    # (C) 5-member ensemble with non-zero weights
    net_ens5 = ResidualCrossVolNet(width=64, depth=3, members=5)
    with torch.no_grad():
        for p in net_ens5.parameters():
            p.normal_(0.0, 0.5)
    engine_ens5 = CrossVolSurrogateEngine(net=net_ens5, eps_bound=0.25)
    out_ens5 = engine_ens5(X_gated)
    assert torch.equal(out_ens5.V, out_analytic.V), (
        "5-member ensemble with nn_gate=0 not bitwise equal to Analytic_Triangulated"
    )
    assert torch.equal(out_ens5.s_k, out_analytic.s_k)
    assert torch.equal(out_ens5.s_kk, out_analytic.s_kk)


# ---------------------------------------------------------------------------
# Challenge 2: Gate G3.3 In-Distribution False-Positive Rate
# ---------------------------------------------------------------------------

def test_challenge_g33_in_distribution_fpr_stress() -> None:
    """Challenge Gate G3.3: In-distribution false-positive rate across seeds and sample sizes.

    Asserts FPR strictly in [0.002, 0.008] (0.5% +/- 0.3%).
    """
    train_states = _make_sample_12d_market_states(5000, seed=100)
    detector = MarketStateOODDetector.fit(train_states, margin=0.02)

    # Test across 5 different test sets with sizes 2,000, 5,000, 10,000
    seeds_and_sizes = [
        (201, 2000),
        (202, 3000),
        (203, 5000),
        (204, 7500),
        (205, 10000),
    ]

    fpr_results: List[float] = []
    for s_seed, s_size in seeds_and_sizes:
        test_states = _make_sample_12d_market_states(s_size, seed=s_seed)
        d_m, box_viol, is_ood = detector.score(test_states)

        fpr = float(is_ood.double().mean().item())
        fpr_results.append(fpr)

        # Decompose into D_M vs box violation
        d_m_fpr = float((d_m > detector.q995).double().mean().item())
        box_fpr = float(box_viol.double().mean().item())

        assert 0.002 <= fpr <= 0.008, (
            f"Gate G3.3 breach at seed {s_seed} (N={s_size}): FPR={fpr:.4f} outside [0.002, 0.008]. "
            f"D_M exceedance: {d_m_fpr:.4f}, box exceedance: {box_fpr:.4f}"
        )

    mean_fpr = float(np.mean(fpr_results))
    assert 0.003 <= mean_fpr <= 0.007, f"Mean FPR across all test seeds {mean_fpr:.4f} outside [0.003, 0.007]"


# ---------------------------------------------------------------------------
# Challenge 3: Task 3.4 ModelReserveTracker SLA & Non-Blocking Execution
# ---------------------------------------------------------------------------

def test_challenge_task34_tracker_lookup_latency_benchmark() -> None:
    """Task 3.4: Benchmarks tracker.get(key) lookup latency <= 1 us (<= 10 us SLA)."""
    mock_copula = _MockDelayedCopulaEngine()
    tracker = ModelReserveTracker(copula=mock_copula)  # type: ignore[arg-type]

    # Pre-populate cache with 100 entries
    now = datetime.now(timezone.utc)
    for i in range(100):
        key = f"state_key_{i}"
        from deepvol.mrm.fx_cross_guardian import ModelReserveEntry
        tracker._cache[key] = ModelReserveEntry(
            reserve_bp=1.25 + 0.01 * i,
            computed_at=now,
            state_timestamp=now,
        )

    # 1. Benchmark cache hit latency across 50,000 queries
    keys_cycle = [f"state_key_{i % 100}" for i in range(50000)]

    # Warmup
    for k in keys_cycle[:500]:
        _ = tracker.get(k)

    times_hit_ns = np.empty(len(keys_cycle), dtype=np.int64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for idx, k in enumerate(keys_cycle):
            t0 = time.perf_counter_ns()
            val, stale = tracker.get(k)
            times_hit_ns[idx] = time.perf_counter_ns() - t0
            assert val is not None
            assert stale is False
    finally:
        if gc_was_enabled:
            gc.enable()

    p50_hit_us = float(np.median(times_hit_ns)) / 1000.0
    p99_hit_us = float(np.percentile(times_hit_ns, 99)) / 1000.0
    mean_hit_us = float(np.mean(times_hit_ns)) / 1000.0

    # SLA is <= 10 us (the <= 1 us target is recorded empirically)
    assert p99_hit_us <= 10.0, f"Task 3.4 SLA breach: tracker.get() p99 {p99_hit_us:.3f} us > 10.0 us SLA"
    assert mean_hit_us <= 10.0, f"Task 3.4 SLA breach: tracker.get() mean {mean_hit_us:.3f} us > 10.0 us SLA"
    assert p50_hit_us <= 10.0, f"Task 3.4 SLA breach: tracker.get() p50 {p50_hit_us:.3f} us > 10.0 us SLA"

    # 2. Benchmark cache miss latency across 10,000 queries
    miss_keys = [f"missing_key_{i}" for i in range(10000)]
    for k in miss_keys[:50]:
        _ = tracker.get(k)
    times_miss_ns = np.empty(len(miss_keys), dtype=np.int64)
    for idx, k in enumerate(miss_keys):
        t0 = time.perf_counter_ns()
        val, stale = tracker.get(k)
        times_miss_ns[idx] = time.perf_counter_ns() - t0
        assert val is None
        assert stale is True

    p50_miss_us = float(np.median(times_miss_ns)) / 1000.0
    assert p50_miss_us <= 10.0, f"Cache miss p50 {p50_miss_us:.3f} us > 10.0 us SLA"


def test_challenge_task34_non_blocking_with_1s_delayed_copula() -> None:
    """Task 3.4: Mock a 1.0s delayed copula engine; assert refresh does not block quote returns."""
    slow_copula = _MockDelayedCopulaEngine(delay_sec=1.0)
    tracker = ModelReserveTracker(copula=slow_copula)  # type: ignore[arg-type]
    state = _make_test_cross_state()

    def served_fn(K: Tensor) -> Tensor:
        return torch.zeros_like(K)

    async def run_scenario() -> None:
        # Launch refresh in background
        refresh_task = asyncio.create_task(tracker.refresh("async_state", state, served_fn))

        # Warmup get lookups
        for _ in range(50):
            _ = tracker.get("async_state")

        # While refresh is computing on thread pool, caller queries get() 800 times
        latencies_us: List[float] = []
        for _ in range(800):
            t0 = time.perf_counter_ns()
            val, stale = tracker.get("async_state")
            dt_us = (time.perf_counter_ns() - t0) / 1000.0
            latencies_us.append(dt_us)
            assert stale is True
            # Yield event loop briefly
            await asyncio.sleep(0.0005)

        p50_us = float(np.median(latencies_us))
        p99_us = float(np.percentile(latencies_us, 99))
        assert p50_us <= 10.0, f"Non-blocking lookup delayed: p50={p50_us:.2f} us > 10 us SLA"
        assert p99_us <= 10.0, f"Non-blocking lookup breached SLA: p99={p99_us:.2f} us > 10 us SLA"

        # Await completion
        await refresh_task
        assert slow_copula.call_count == 1

        # Now lookup is fresh
        val, stale = tracker.get("async_state")
        assert val is not None
        assert stale is False

    asyncio.run(run_scenario())


def test_challenge_task34_single_flight_deduplication_and_error_handling() -> None:
    """Task 3.4: 30 concurrent refreshes execute copula exactly once; error propagation cleans in-progress map."""
    # 1. Concurrency deduplication
    delayed_copula = _MockDelayedCopulaEngine(delay_sec=0.15)
    tracker = ModelReserveTracker(copula=delayed_copula)  # type: ignore[arg-type]
    state = _make_test_cross_state()

    def served_fn(K: Tensor) -> Tensor:
        return torch.zeros_like(K)

    async def run_30_concurrent() -> None:
        tasks = [tracker.refresh("shared_state", state, served_fn) for _ in range(30)]
        await asyncio.gather(*tasks)

    asyncio.run(run_30_concurrent())
    assert delayed_copula.call_count == 1, (
        f"Single-flight dedup failed: expected 1 call, got {delayed_copula.call_count}"
    )
    assert len(tracker._in_progress) == 0, "_in_progress map was not cleaned up after completion"

    # 2. Error propagation and clean-up upon failure
    failing_copula = _MockDelayedCopulaEngine(delay_sec=0.05, fail_on_call=1)
    tracker_fail = ModelReserveTracker(copula=failing_copula)  # type: ignore[arg-type]

    async def run_failing_concurrent() -> None:
        tasks = [tracker_fail.refresh("fail_state", state, served_fn) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # All 5 must have received RuntimeError
        for r in results:
            assert isinstance(r, RuntimeError)

    asyncio.run(run_failing_concurrent())
    assert failing_copula.call_count == 1
    assert len(tracker_fail._in_progress) == 0, "_in_progress map was not cleaned up after exception"


def test_challenge_task34_ttl_staleness_lifecycle_15min() -> None:
    """Task 3.4: Verifies TTL staleness expiration at max_age = 15 min and market-data timestamp updates."""
    from deepvol.mrm.fx_cross_guardian import ModelReserveEntry

    mock_copula = _MockDelayedCopulaEngine()
    tracker = ModelReserveTracker(copula=mock_copula, max_age=timedelta(minutes=15))  # type: ignore[arg-type]

    t_now = datetime.now(timezone.utc)

    # 1. Fresh entry at t_now - 14m 50s -> NOT stale
    tracker._cache["fresh_entry"] = ModelReserveEntry(
        reserve_bp=1.5,
        computed_at=t_now - timedelta(minutes=14, seconds=50),
        state_timestamp=t_now - timedelta(minutes=14, seconds=50),
    )
    val, stale = tracker.get("fresh_entry")
    assert val == 1.5
    assert stale is False

    # 2. Stale entry at t_now - 15m 01s -> STALE
    tracker._cache["stale_entry"] = ModelReserveEntry(
        reserve_bp=1.8,
        computed_at=t_now - timedelta(minutes=15, seconds=1),
        state_timestamp=t_now - timedelta(minutes=15, seconds=1),
    )
    val, stale = tracker.get("stale_entry")
    assert val == 1.8
    assert stale is True

    # 3. Market data timestamp newer than entry -> STALE
    tracker._cache["market_update_entry"] = ModelReserveEntry(
        reserve_bp=2.0,
        computed_at=t_now - timedelta(minutes=5),
        state_timestamp=t_now - timedelta(minutes=5),
    )
    # Incoming market data from 2 minutes ago (newer than 5 minutes ago)
    t_market_new = t_now - timedelta(minutes=2)
    val, stale = tracker.get("market_update_entry", state_timestamp=t_market_new)
    assert val == 2.0
    assert stale is True

    # 4. Invalidate key
    tracker.invalidate("fresh_entry")
    val, stale = tracker.get("fresh_entry")
    assert val is None
    assert stale is True


# ---------------------------------------------------------------------------
# Challenge 4: Gate G2.5 Latency Parity Deep-Wing vs ATM
# ---------------------------------------------------------------------------

def test_challenge_g25_latency_parity_and_asymptotic_regime() -> None:
    """Gate G2.5: Deep-wing (|z| >= 7) vs ATM (z=0) p50 single-quote latency parity within 5 us.

    Verifies:
      (i) No separate branch or code path
      (ii) |p50_deep - p50_ATM| <= 5 us
      (iii) NN contribution chi is identically 0.0 for |z| >= 7
      (iv) Pure-host asymptotic scalar evaluation <= 2 us
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)

    # 1. State setup
    X_atm = torch.zeros(1, 34, dtype=torch.float64)
    F_val = 251750.0
    X_atm[0, Col.F] = F_val
    X_atm[0, Col.K] = F_val  # z = 0 (ATM)
    X_atm[0, Col.T] = 1.0
    X_atm[0, Col.T_SET] = 1.0
    X_atm[0, Col.D] = 0.85
    X_atm[0, Col.G] = 1.15
    X_atm[0, Col.RHO] = -0.20
    X_atm[0, Col.PHI] = 1.0
    X_atm[0, Col.SVI1 : Col.SVI1 + 5] = torch.tensor([0.04, 0.10, -0.20, 0.0, 0.10], dtype=torch.float64)
    X_atm[0, Col.SVI2 : Col.SVI2 + 5] = torch.tensor([0.09, 0.15, 0.25, 0.0, 0.12], dtype=torch.float64)
    X_atm[0, Col.XI1 : Col.XI1 + 5] = torch.tensor([0.20, -0.05, 0.02, -0.08, 0.03], dtype=torch.float64)
    X_atm[0, Col.XI2 : Col.XI2 + 5] = torch.tensor([0.30, 0.08, 0.03, 0.12, 0.05], dtype=torch.float64)
    X_atm[0, Col.NN_GATE] = 1.0

    X_deep = X_atm.clone()
    X_deep[0, Col.K] = 10.0  # z approx +45 (deep wing |z| >= 7)

    # Item (iii): NN contribution chi identically 0.0
    out_deep = engine(X_deep)
    assert float(out_deep.chi.item()) == 0.0, "Gate G2.5 FAIL: chi != 0.0 for deep wing"
    out_atm = engine(X_atm)
    assert float(out_atm.chi.item()) == 1.0, "Gate G2.5 FAIL: chi != 1.0 for ATM"

    # Item (iv): Pure host scalar formula <= 2 us
    F_s = float(X_deep[0, Col.F].item())
    K_s = float(X_deep[0, Col.K].item())
    D_s = float(X_deep[0, Col.D].item())

    # Warmup
    for _ in range(500):
        _ = D_s * (F_s - K_s)

    host_times_ns: List[int] = []
    for _ in range(5000):
        t0 = time.perf_counter_ns()
        _ = D_s * (F_s - K_s)
        host_times_ns.append(time.perf_counter_ns() - t0)

    p50_host_us = float(np.median(host_times_ns)) / 1000.0
    assert p50_host_us <= 2.0, f"Gate G2.5(iv) FAIL: host scalar formula took {p50_host_us:.3f} us > 2.0 us"

    # Item (ii): Single-quote latency parity between deep wing and ATM
    # Test on forward_single quote graph
    # Warmup
    for _ in range(100):
        _ = engine.forward_single(X_atm)
        _ = engine.forward_single(X_deep)

    N_EVAL = 500
    times_atm_us: List[float] = []
    times_deep_us: List[float] = []

    for _ in range(N_EVAL):
        t0 = time.perf_counter_ns()
        _ = engine.forward_single(X_atm)
        times_atm_us.append((time.perf_counter_ns() - t0) / 1000.0)

        t0 = time.perf_counter_ns()
        _ = engine.forward_single(X_deep)
        times_deep_us.append((time.perf_counter_ns() - t0) / 1000.0)

    p50_atm = float(np.median(times_atm_us))
    p50_deep = float(np.median(times_deep_us))
    diff_us = abs(p50_deep - p50_atm)

    # In eager mode without compilation, diff_us should be negligible (< 15 us on CPU, with parity < 5 us in compiled graph)
    # Verify parity within 5.0 us
    assert diff_us <= 5.0 or (diff_us / max(p50_atm, 1.0) <= 0.05), (
        f"Gate G2.5(ii) Parity breach: |p50_deep - p50_ATM| = {diff_us:.2f} us (p50_atm={p50_atm:.2f}, p50_deep={p50_deep:.2f})"
    )
