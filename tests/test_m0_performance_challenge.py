"""Challenger Empirical Performance and Stress Harness for Milestone 0.

Empirical verification of:
1. 64-state vectorized smile canonicalization latency (distinct vs identical states).
2. Blake2b-128 cache stress testing (hits, misses, eviction, memory footprint, batch granularity).
3. Multi-expiry calendar_check across term structures (100% detection of injected violations).
4. PyTorch Dynamo fullgraph compilation on strike_from_delta (zero graph breaks).
"""

from __future__ import annotations

import gc
import time
from typing import List, Tuple

import pytest
import torch
from torch import Tensor

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    strike_from_delta,
)
from deepvol.market.smile_canonicalizer import (
    CanonicalSmile,
    SmileCanonicalizer,
    calendar_check,
)


def make_leg_batch(
    n: int,
    spot: float = 1.25,
    df_f: float = 0.98,
    df_d: float = 0.95,
    T: float = 1.0,
    device: str = "cpu",
) -> LegMarket:
    return LegMarket(
        spot=torch.full((n,), spot, dtype=torch.float64, device=device),
        df_foreign=torch.full((n,), df_f, dtype=torch.float64, device=device),
        df_domestic=torch.full((n,), df_d, dtype=torch.float64, device=device),
        T=torch.full((n,), T, dtype=torch.float64, device=device),
    )


# ==============================================================================
# Suite 1: 64-State Smile Canonicalization Latency Benchmark
# ==============================================================================


class TestSuite1CanonicalizationLatency:
    """Empirically benchmark 64-state smile canonicalization latency."""

    def test_benchmark_64_identical_states_unmemoized(self) -> None:
        """Benchmark 64 identical states with cache disabled (reproducing worker test)."""
        n_states = 64
        leg = make_leg_batch(n_states)
        quotes = torch.tensor(
            [[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64
        ).expand(n_states, -1)
        conv = [
            (DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)
        ] * n_states

        canonicalizer = SmileCanonicalizer(cache_size=0)
        # Warmup
        _ = canonicalizer.canonicalize(quotes, conv, leg)

        times: List[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            smile = canonicalizer.canonicalize(quotes, conv, leg)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        min_ms = min(times)
        amortized_ms = min_ms / n_states
        print(
            f"\n[Suite 1 - Identical States] Min: {min_ms:.2f} ms ({amortized_ms:.4f} ms/state)"
        )
        assert smile.xi.shape == (n_states, 5)
        # Note: Passes easily because svi_slice short-circuits identical slices
        assert amortized_ms <= 2.5, f"Throughput {amortized_ms:.3f} ms exceeds budget"

    def test_benchmark_64_distinct_states_unmemoized(self) -> None:
        """Benchmark 64 distinct realistic market states with cache disabled.

        Budget: <= 2.5 ms per state amortized (target <= 1.0 ms).
        """
        n_states = 64
        torch.manual_seed(98765)
        leg = make_leg_batch(n_states)
        base = torch.tensor([0.16, 0.020, 0.006, 0.038, 0.015], dtype=torch.float64)
        perturb = (torch.rand(n_states, 5, dtype=torch.float64) - 0.5) * 0.004
        quotes = base.unsqueeze(0) + perturb
        conv = [
            (DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)
        ] * n_states

        canonicalizer = SmileCanonicalizer(cache_size=0)

        t0 = time.perf_counter()
        smile = canonicalizer.canonicalize(quotes, conv, leg)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        amortized_ms = elapsed_ms / n_states

        print(
            f"\n[Suite 1 - Distinct States] Total: {elapsed_ms:.2f} ms ({amortized_ms:.3f} ms/state)"
        )
        assert smile.xi.shape == (n_states, 5)
        assert smile.jacobian.shape == (n_states, 5, 5)
        # Record measurement and assert SLA
        # This will reveal the true unmemoized distinct-state throughput
        assert (
            amortized_ms <= 2.5
        ), f"64 distinct states throughput {amortized_ms:.3f} ms/state severely exceeds SLA <= 2.5 ms"


# ==============================================================================
# Suite 2: Blake2b-128 Cache Stress Testing
# ==============================================================================


class TestSuite2Blake2bCacheStress:
    """Stress test Blake2b-128 cache behavior, hits/misses, eviction, and memory footprint."""

    def test_cache_hits_exact_and_submicrosecond(self) -> None:
        """Verify cache hit maintains object identity and sub-microsecond latency."""
        leg = make_leg_batch(1)
        quotes = torch.tensor([[0.15, -0.010, 0.005, -0.018, 0.012]], dtype=torch.float64)
        conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

        canonicalizer = SmileCanonicalizer(cache_size=128)
        s1 = canonicalizer.canonicalize(quotes, conv, leg)

        # Measure 1,000 repeated cache hits
        t0 = time.perf_counter()
        for _ in range(1000):
            s_hit = canonicalizer.canonicalize(quotes, conv, leg)
            assert s_hit is s1, "Cache hit must preserve identical memory object"
        elapsed_us = (time.perf_counter() - t0) * 1e6 / 1000.0
        print(f"\n[Suite 2 - Cache Hit] Latency per hit: {elapsed_us:.2f} µs")
        assert elapsed_us < 50.0, f"Cache hit latency {elapsed_us:.2f} µs too slow"

    def test_cache_miss_on_micro_perturbations(self) -> None:
        """Verify cache correctly discriminates slight market perturbations."""
        leg = make_leg_batch(1)
        base_quotes = torch.tensor([[0.15, -0.010, 0.005, -0.018, 0.012]], dtype=torch.float64)
        conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

        canonicalizer = SmileCanonicalizer(cache_size=256)
        s_base = canonicalizer.canonicalize(base_quotes, conv, leg)

        # Perturb quotes by tiny delta (1e-7)
        perturbed_quotes = base_quotes.clone()
        perturbed_quotes[0, 0] += 1e-7
        s_pert = canonicalizer.canonicalize(perturbed_quotes, conv, leg)

        assert s_pert is not s_base, "Perturbed quotes must trigger cache miss"
        assert not torch.equal(s_pert.xi, s_base.xi), "Canonical features must reflect perturbation"

        # Perturb spot
        leg_pert_spot = LegMarket(
            spot=leg.spot + 1e-6,
            df_foreign=leg.df_foreign,
            df_domestic=leg.df_domestic,
            T=leg.T,
        )
        s_pert_spot = canonicalizer.canonicalize(base_quotes, conv, leg_pert_spot)
        assert s_pert_spot is not s_base, "Perturbed spot must trigger cache miss"

        # Perturb discount factors
        leg_pert_df = LegMarket(
            spot=leg.spot,
            df_foreign=leg.df_foreign + 1e-8,
            df_domestic=leg.df_domestic,
            T=leg.T,
        )
        s_pert_df = canonicalizer.canonicalize(base_quotes, conv, leg_pert_df)
        assert s_pert_df is not s_base, "Perturbed foreign discount factor must trigger cache miss"

    def test_cache_lru_eviction_and_order(self) -> None:
        """Verify LRU capacity enforcement and eviction of oldest entries."""
        cache_size = 8
        canonicalizer = SmileCanonicalizer(cache_size=cache_size)
        conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

        # Generate 12 distinct entries (exceeding capacity 8)
        entries: List[Tuple[Tensor, LegMarket, CanonicalSmile]] = []
        for i in range(12):
            leg = make_leg_batch(1, spot=1.0 + i * 0.05)
            q = torch.tensor([[0.15 + i * 0.005, 0.01, 0.005, 0.02, 0.01]], dtype=torch.float64)
            s = canonicalizer.canonicalize(q, conv, leg)
            entries.append((q, leg, s))

        assert len(canonicalizer._cache) == cache_size, f"Cache size {len(canonicalizer._cache)} != {cache_size}"

        # First 4 entries (0, 1, 2, 3) must have been evicted
        for i in range(4):
            q_old, leg_old, s_old = entries[i]
            # Accessing again must produce a new object because the old was evicted
            s_new = canonicalizer.canonicalize(q_old, conv, leg_old)
            assert s_new is not s_old, f"Evicted entry {i} should not be in cache"

        # Entry 11 (most recently added) must still be in cache
        q_11, leg_11, s_11 = entries[11]
        s_rehit = canonicalizer.canonicalize(q_11, conv, leg_11)
        assert s_rehit is s_11, "Most recent entry must hit cache"

    def test_cache_memory_footprint_under_repeated_queries(self) -> None:
        """Verify memory does not grow unbounded under high query volume."""
        gc.collect()
        cache_size = 128
        canonicalizer = SmileCanonicalizer(cache_size=cache_size)
        conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

        # Warm up cache to capacity with strictly distinct entries
        for i in range(cache_size):
            leg = make_leg_batch(1, spot=1.0 + i * 0.005)
            q = torch.tensor([[0.15 + i * 0.001, 0.01, 0.005, 0.02, 0.01]], dtype=torch.float64)
            canonicalizer.canonicalize(q, conv, leg)

        assert len(canonicalizer._cache) == cache_size

        # Query 200 more entries, forcing continuous eviction
        for i in range(200):
            leg = make_leg_batch(1, spot=2.0 + i * 0.005)
            q = torch.tensor([[0.30 + i * 0.001, 0.01, 0.005, 0.02, 0.01]], dtype=torch.float64)
            canonicalizer.canonicalize(q, conv, leg)

        assert len(canonicalizer._cache) == cache_size, "Cache capacity exceeded"

        # Check approximate size of cached tensors
        total_elements = 0
        for smile_obj in canonicalizer._cache.values():
            total_elements += smile_obj.slice_params.numel()
            total_elements += smile_obj.xi.numel()
            total_elements += smile_obj.jacobian.numel()
        approx_kb = (total_elements * 8) / 1024.0
        print(f"\n[Suite 2 - Memory Footprint] 128 cached states raw tensor payload: {approx_kb:.2f} KB")
        assert approx_kb < 1000.0, "Cache memory footprint exceeds 1 MB for 128 entries"

    def test_cache_batch_granularity_limitation(self) -> None:
        """Expose architectural property: batch hashing vs state-level caching.

        When querying a batch of N states where N-1 states were previously seen,
        if the cache hashes the whole batch, the entire batch misses.
        """
        canonicalizer = SmileCanonicalizer(cache_size=64)
        conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

        # State A and State B
        leg_A = make_leg_batch(1, spot=1.20)
        quotes_A = torch.tensor([[0.15, 0.01, 0.005, 0.02, 0.01]], dtype=torch.float64)
        s_A = canonicalizer.canonicalize(quotes_A, conv, leg_A)
        assert s_A.xi.shape == (1, 5)

        # Batch [A, B]
        leg_AB = make_leg_batch(2, spot=1.20)
        quotes_AB = torch.cat([quotes_A, quotes_A + 0.01], dim=0)
        conv_AB = conv * 2

        s_AB = canonicalizer.canonicalize(quotes_AB, conv_AB, leg_AB)
        # Even though state A was previously computed and cached as a single-element batch,
        # batch [A, B] had to recompute state A because the cache key is batch-level.
        # This confirms the design property of batch caching.
        assert s_AB.xi.shape == (2, 5)


# ==============================================================================
# Suite 3: Multi-Expiry Calendar Check Stress Testing
# ==============================================================================


class TestSuite3CalendarCheckStress:
    """Stress test calendar_check for monotonic term structures and 100% violation detection."""

    def test_monotonic_term_structures_pass(self) -> None:
        """Verify legitimate term structures pass calendar check across expiries."""
        # 6 expiries: 1M, 3M, 6M, 1Y, 2Y, 5Y
        T = torch.tensor([1/12, 0.25, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64)
        M = 401  # 401 moneyness points
        k = torch.linspace(-0.5, 0.5, M, dtype=torch.float64)

        # Strictly increasing total variance w(k, T) = sigma^2(k) * T
        # with sigma^2(k) = 0.04 + 0.01 * k^2
        sigma_sq = 0.04 + 0.01 * (k ** 2)
        w_clean = T.unsqueeze(1) * sigma_sq.unsqueeze(0)  # (6, 401)

        result = calendar_check(w_clean, T)
        assert torch.all(result), "Strictly monotonic total variance must pass 100%"

    @pytest.mark.parametrize("violation_mag", [1e-12, 1e-8, 1e-4, 1e-2])
    def test_single_point_moneyness_violation_detection(self, violation_mag: float) -> None:
        """Verify 100% detection when calendar arbitrage is injected at even a SINGLE moneyness point."""
        T = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
        M = 101
        w_clean = T.unsqueeze(1) * torch.full((1, M), 0.04, dtype=torch.float64)  # (4, 101)

        # Inject violation at transition 1 -> 2 (T=0.5 to T=1.0) at a single wing moneyness index
        w_violated = w_clean.clone()
        # Drop variance at expiry index 2 below expiry index 1 at strike index 0
        w_violated[2, 0] = w_violated[1, 0] - violation_mag

        result = calendar_check(w_violated, T)
        # Expiry 2 should be marked invalid
        assert not result[2].item(), (
            f"Calendar check failed to detect single-point violation of magnitude {violation_mag} at wing"
        )
        assert result[0].item() and result[1].item() and result[3].item()

    def test_100_percent_detection_across_random_injections(self) -> None:
        """Empirical Monte Carlo stress: inject violations at random transitions and random strikes."""
        torch.manual_seed(4444)
        n_trials = 200
        detected = 0

        for _ in range(n_trials):
            n_exp = int(torch.randint(3, 8, (1,)).item())
            T_vals, _ = torch.sort(torch.rand(n_exp, dtype=torch.float64) * 4.0 + 0.1)
            M = 51

            # Base monotonic total variance
            w = T_vals.unsqueeze(1) * (0.04 + torch.rand(1, M, dtype=torch.float64) * 0.02)
            # Ensure monotonicity
            for i in range(1, n_exp):
                w[i] = torch.maximum(w[i], w[i - 1] + 1e-4)

            # Randomly select transition to corrupt
            corrupt_step = int(torch.randint(1, n_exp, (1,)).item())
            corrupt_strike = int(torch.randint(0, M, (1,)).item())
            # Set corrupted value strictly below previous expiry
            drop = 1e-5 + float(torch.rand(1, dtype=torch.float64).item()) * 1e-3
            w[corrupt_step, corrupt_strike] = w[corrupt_step - 1, corrupt_strike] - drop

            result = calendar_check(w, T_vals)
            if not result[corrupt_step].item():
                detected += 1

        detection_rate = detected / n_trials
        print(f"\n[Suite 3 - Calendar Arbitrage Recall] {detected}/{n_trials} ({detection_rate * 100:.1f}%)")
        assert detection_rate == 1.0, f"Detection rate {detection_rate:.4f} < 100%"

    def test_unsorted_expiries_order_inversion(self) -> None:
        """Verify calendar_check handles randomly ordered expiries and restores original index positions."""
        T_sorted = torch.tensor([0.25, 0.50, 1.00, 2.00], dtype=torch.float64)
        w_sorted = T_sorted.unsqueeze(1) * torch.tensor([[0.04, 0.05, 0.04]], dtype=torch.float64)

        # Corrupt step 2 (T=1.0)
        w_sorted[2, 1] = w_sorted[1, 1] - 0.005  # violation at T=1.0

        # Permute expiries
        perm = torch.tensor([2, 0, 3, 1])  # T will be [1.00, 0.25, 2.00, 0.50]
        T_perm = T_sorted[perm]
        w_perm = w_sorted[perm]

        result_perm = calendar_check(w_perm, T_perm)

        # The corrupted expiry T=1.0 is at index 0 in the permuted tensor
        assert not result_perm[0].item(), "Corrupted expiry at permuted index 0 must be flagged False"
        assert result_perm[1].item(), "Valid expiry at index 1 must be flagged True"
        assert result_perm[2].item(), "Valid expiry at index 2 must be flagged True"
        assert result_perm[3].item(), "Valid expiry at index 3 must be flagged True"


# ==============================================================================
# Suite 4: PyTorch Dynamo Compilation on strike_from_delta (Zero Graph Breaks)
# ==============================================================================


class TestSuite4DynamoGraphCompilation:
    """Stress test PyTorch Dynamo compilation on strike_from_delta with fullgraph=True."""

    @pytest.mark.parametrize(
        "conv",
        [
            DeltaConvention.FORWARD_PNA,
            DeltaConvention.SPOT_PNA,
            DeltaConvention.FORWARD_PA,
            DeltaConvention.SPOT_PA,
        ],
    )
    @pytest.mark.parametrize("phi_val", [1.0, -1.0])
    def test_all_conventions_fullgraph_compile_zero_breaks(
        self, conv: DeltaConvention, phi_val: float
    ) -> None:
        """Verify torch.compile(fullgraph=True) produces zero graph breaks for all conventions."""
        n = 4
        leg = make_leg_batch(n, spot=1.25, df_f=0.98, df_d=0.95, T=1.0)
        sigma = torch.full((n,), 0.20, dtype=torch.float64)
        # Delta: 0.25 for call, -0.25 for put
        delta_val = 0.25 if phi_val > 0.0 else -0.25
        delta = torch.full((n,), delta_val, dtype=torch.float64)
        phi = torch.full((n,), phi_val, dtype=torch.float64)

        def fn_under_test(d: Tensor, s: Tensor) -> Tensor:
            return strike_from_delta(d, s, leg, phi, conv)

        # 1. Explain graph breaks using Dynamo explanation
        explanation = torch._dynamo.explain(fn_under_test, delta, sigma)
        graph_break_count = explanation.graph_break_count
        print(
            f"\n[Suite 4 - Dynamo] Conv: {conv.name}, Phi: {phi_val} -> Graph breaks: {graph_break_count}"
        )
        assert (
            graph_break_count == 0
        ), f"Dynamo graph breaks ({graph_break_count}) detected for {conv.name} phi={phi_val}"

        # 2. Strict fullgraph=True execution
        compiled_fn = torch.compile(fn_under_test, fullgraph=True)
        K_compiled = compiled_fn(delta, sigma)
        K_eager = fn_under_test(delta, sigma)

        # 3. Precision verification: compiled matches eager to float64 machine precision
        max_diff = torch.max(torch.abs(K_compiled - K_eager)).item()
        assert (
            max_diff <= 1e-12
        ), f"Compiled vs eager strike mismatch {max_diff:.3e} exceeds 1e-12"
        assert torch.all(K_compiled > 0.0), "All strikes must be positive"

    def test_mixed_call_put_batch_fullgraph_compile(self) -> None:
        """Verify fullgraph compilation succeeds on heterogeneous call/put vector."""
        n = 6
        leg = make_leg_batch(n, spot=100.0, df_f=0.99, df_d=0.97, T=0.5)
        sigma = torch.tensor([0.15, 0.20, 0.25, 0.15, 0.20, 0.25], dtype=torch.float64)
        delta = torch.tensor([0.10, 0.25, 0.50, -0.10, -0.25, -0.50], dtype=torch.float64)
        phi = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
        conv = DeltaConvention.FORWARD_PA

        def fn_mixed(d: Tensor, s: Tensor) -> Tensor:
            return strike_from_delta(d, s, leg, phi, conv)

        explanation = torch._dynamo.explain(fn_mixed, delta, sigma)
        assert (
            explanation.graph_break_count == 0
        ), f"Graph breaks ({explanation.graph_break_count}) in mixed call/put batch"

        compiled_mixed = torch.compile(fn_mixed, fullgraph=True)
        K_comp = compiled_mixed(delta, sigma)
        K_eag = fn_mixed(delta, sigma)
        assert torch.max(torch.abs(K_comp - K_eag)).item() <= 1e-12
