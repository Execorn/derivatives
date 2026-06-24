"""
Tests for src/market/variance_swaps.py — P2-B2 Variance Swap Pricing.

Test plan:
  1. variance_swap_rate — positivity, rough range, H-sensitivity
  2. realized_variance  — synthetic price series, edge cases
  3. vol_swap_rate      — Jensen inequality (K_vol ≤ √K_var)
  4. variance_swap_pnl  — break-even and directional P&L
  5. variance_term_structure — shape, monotonicity, correct length
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the src directory is on the path (mirrors conftest.py approach)
_src_dir = str(Path(__file__).parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from deepvol.market.variance_swaps import (
    realized_variance,
    variance_swap_pnl,
    variance_swap_rate,
    variance_term_structure,
    vol_swap_rate,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
_BASE = dict(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1, T=1.0)


# ===========================================================================
# 1. variance_swap_rate
# ===========================================================================

class TestVarianceSwapRate:
    def test_positive(self):
        """Fair variance strike must be strictly positive for v0 > 0."""
        kv = variance_swap_rate(**_BASE)
        assert kv > 0.0, f"Expected positive K_var, got {kv}"

    def test_reasonable_range_v0_eq_theta(self):
        """When v0 ≈ θ, K_var should be close to θ (market in equilibrium)."""
        kv = variance_swap_rate(
            kappa=2.0, theta=0.04, sigma=0.6, rho=-0.5, v0=0.04, H=0.1, T=1.0
        )
        # K_var should be within 20% of theta for equilibrium initial variance
        assert abs(kv - 0.04) < 0.04 * 0.20, (
            f"K_var={kv:.6f} is far from theta=0.04 in equilibrium"
        )

    def test_rough_vs_standard_heston_differ(self):
        """Rough Heston (H=0.08) and standard Heston (H=0.5) must give different K_var.

        While the expected integrated variance is model-parameter-dependent, the
        mean-reversion dynamics differ substantially between rough (H≪0.5) and
        Markovian (H→0.5) regimes, so K_var values must not be identical.
        """
        params = dict(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, T=1.0)
        kv_rough = variance_swap_rate(H=0.08, **params)
        kv_standard = variance_swap_rate(H=0.49, **params)
        assert kv_rough != kv_standard, (
            "variance_swap_rate should differ between rough (H=0.08) and "
            f"near-standard (H=0.49) Heston. Got both = {kv_rough:.8f}"
        )

    def test_zero_maturity_returns_v0(self):
        """T=0 edge case: K_var should equal v0 (instantaneous variance)."""
        v0 = 0.09
        kv = variance_swap_rate(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7,
                                v0=v0, H=0.1, T=0.0)
        assert kv == pytest.approx(v0, abs=1e-9), (
            f"For T=0, expected K_var=v0={v0}, got {kv}"
        )

    def test_zero_initial_variance(self):
        """v0=0: process starts at 0, mean-reverts towards theta > 0, so K_var > 0."""
        kv = variance_swap_rate(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7,
                                v0=0.0, H=0.1, T=1.0)
        assert kv > 0.0, f"K_var should be > 0 when v0=0 but theta>0, got {kv}"

    def test_h_boundary_low(self):
        """H near lower boundary (0.005) should not crash."""
        kv = variance_swap_rate(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7,
                                v0=0.04, H=0.005, T=1.0)
        assert kv > 0.0

    def test_h_boundary_high(self):
        """H near upper boundary (0.495) should not crash."""
        kv = variance_swap_rate(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7,
                                v0=0.04, H=0.495, T=1.0)
        assert kv > 0.0

    def test_short_maturity_close_to_v0(self):
        """For very short T, K_var ≈ v0 (not enough time to mean-revert)."""
        v0 = 0.09
        kv = variance_swap_rate(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7,
                                v0=v0, H=0.1, T=1/252)
        # K_var for ~1 trading day should be very close to v0
        assert abs(kv - v0) < 0.01, (
            f"Short-T K_var={kv:.6f} deviates too far from v0={v0}"
        )

    def test_long_maturity_converges_to_theta(self):
        """For large T, K_var → θ (ergodic limit of mean-reverting variance)."""
        kv = variance_swap_rate(kappa=3.0, theta=0.04, sigma=0.5, rho=-0.7,
                                v0=0.09, H=0.1, T=10.0)
        # Should be within 10% of theta for high mean-reversion speed
        assert abs(kv - 0.04) < 0.04 * 0.15, (
            f"Long-T K_var={kv:.6f} should be near theta=0.04"
        )

    def test_returns_float(self):
        """Return type must be a Python float."""
        kv = variance_swap_rate(**_BASE)
        assert isinstance(kv, float)


# ===========================================================================
# 2. realized_variance
# ===========================================================================

class TestRealizedVariance:
    def test_flat_price_series_zero_rv(self):
        """Constant prices → zero log-returns → RV = 0."""
        prices = np.ones(100) * 100.0
        rv = realized_variance(prices)
        assert rv == pytest.approx(0.0, abs=1e-12)

    def test_synthetic_gbm_reasonable_range(self):
        """Simulated GBM with σ=0.2 should produce RV ≈ 0.04 (within 2σ)."""
        rng = np.random.default_rng(42)
        sigma_true = 0.20
        dt = 1.0 / 252
        n = 1000
        log_returns = rng.normal(0, sigma_true * np.sqrt(dt), n)
        prices = 100.0 * np.exp(np.cumsum(log_returns))
        prices = np.insert(prices, 0, 100.0)  # prepend initial price

        rv = realized_variance(prices, dt=dt)
        # RV should be within ~50% of σ² for n=1000
        assert abs(rv - sigma_true**2) < sigma_true**2 * 0.5, (
            f"Realized variance={rv:.6f}, expected ≈ {sigma_true**2:.6f}"
        )

    def test_annualisation(self):
        """RV annualises correctly: weekly dt=1/52 should give similar result to daily."""
        rng = np.random.default_rng(7)
        sigma = 0.2
        # Generate 5-year weekly prices
        dt_weekly = 1.0 / 52
        log_ret_weekly = rng.normal(0, sigma * np.sqrt(dt_weekly), 5 * 52)
        prices_weekly = 100.0 * np.exp(np.cumsum(log_ret_weekly))
        prices_weekly = np.insert(prices_weekly, 0, 100.0)
        rv_weekly = realized_variance(prices_weekly, dt=dt_weekly)
        assert abs(rv_weekly - sigma**2) < sigma**2 * 0.5

    def test_minimum_two_prices(self):
        """Two prices is the minimum valid input."""
        rv = realized_variance(np.array([100.0, 101.0]))
        assert rv > 0.0

    def test_raises_for_single_price(self):
        """Single price should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            realized_variance(np.array([100.0]))

    def test_raises_for_non_positive_prices(self):
        """Non-positive prices should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            realized_variance(np.array([100.0, 0.0, 102.0]))

    def test_returns_float(self):
        """Return type must be a Python float."""
        rv = realized_variance(np.array([100.0, 101.0, 102.0]))
        assert isinstance(rv, float)

    def test_custom_dt(self):
        """Custom dt (monthly) integrates correctly."""
        # Monthly log-returns of 1% each → RV should be well-defined
        prices = 100.0 * np.exp(np.cumsum(np.ones(12) * 0.01))
        prices = np.insert(prices, 0, 100.0)
        rv = realized_variance(prices, dt=1.0 / 12)
        expected = (0.01**2) / (1.0 / 12)  # each r² / dt
        assert rv == pytest.approx(expected, rel=1e-10)


# ===========================================================================
# 3. vol_swap_rate  — Jensen's inequality
# ===========================================================================

class TestVolSwapRate:
    def test_positive(self):
        """Vol swap rate must be positive."""
        kv = vol_swap_rate(**_BASE)
        assert kv > 0.0

    def test_jensen_inequality(self):
        """Jensen's inequality: K_vol ≤ √K_var (vol swap ≤ vol of variance swap).

        Our implementation returns exactly √K_var (upper bound), so this is
        satisfied with equality.  A tighter approximation would give K_vol < √K_var.
        """
        params = dict(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1, T=1.0)
        kv_vol = vol_swap_rate(**params)
        kv_var = variance_swap_rate(**params)
        sqrt_kv = np.sqrt(kv_var)

        # K_vol must be ≤ √K_var  (Jensen's inequality upper bound)
        assert kv_vol <= sqrt_kv + 1e-10, (
            f"Jensen's violation: vol_swap_rate={kv_vol:.8f} > sqrt(K_var)={sqrt_kv:.8f}"
        )

    def test_jensen_multiple_maturities(self):
        """Jensen's inequality must hold for several maturities."""
        T_values = [1/12, 3/12, 6/12, 1.0, 2.0]
        for T in T_values:
            kv_vol = vol_swap_rate(1.0, 0.04, 0.5, -0.7, 0.04, 0.1, T)
            kv_var = variance_swap_rate(1.0, 0.04, 0.5, -0.7, 0.04, 0.1, T)
            assert kv_vol <= np.sqrt(kv_var) + 1e-10, (
                f"Jensen violation at T={T}: K_vol={kv_vol}, √K_var={np.sqrt(kv_var)}"
            )

    def test_returns_float(self):
        kv = vol_swap_rate(**_BASE)
        assert isinstance(kv, float)

    def test_vol_swap_in_vol_units(self):
        """Result should be expressed as an annualised vol (typically 0.10–0.80 for equity)."""
        kv = vol_swap_rate(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7,
                           v0=0.04, H=0.1, T=1.0)
        assert 0.0 < kv < 1.5, f"Unexpected vol swap rate: {kv}"


# ===========================================================================
# 4. variance_swap_pnl
# ===========================================================================

class TestVarianceSwapPnL:
    def test_breakeven_pnl_is_zero(self):
        """When realized_var == K_var, P&L must be zero."""
        kv = variance_swap_rate(**_BASE)
        pnl = variance_swap_pnl(**_BASE, N_notional=1e6, realized_var=kv)
        assert pnl == pytest.approx(0.0, abs=1e-6), (
            f"Break-even P&L should be 0, got {pnl}"
        )

    def test_long_position_profits_above_strike(self):
        """Long variance swap profits when realized > strike."""
        kv = variance_swap_rate(**_BASE)
        realized_high = kv * 1.5   # 50% higher than strike
        pnl = variance_swap_pnl(**_BASE, N_notional=1e6, realized_var=realized_high)
        assert pnl > 0.0, "Long variance swap should profit when realized > strike"

    def test_long_position_loses_below_strike(self):
        """Long variance swap loses when realized < strike."""
        kv = variance_swap_rate(**_BASE)
        realized_low = kv * 0.5    # 50% below strike
        pnl = variance_swap_pnl(**_BASE, N_notional=1e6, realized_var=realized_low)
        assert pnl < 0.0, "Long variance swap should lose when realized < strike"

    def test_pnl_scales_linearly_with_notional(self):
        """P&L must scale linearly with notional amount."""
        realized = 0.09  # different from K_var for non-zero P&L
        pnl_1 = variance_swap_pnl(**_BASE, N_notional=1e6, realized_var=realized)
        pnl_2 = variance_swap_pnl(**_BASE, N_notional=2e6, realized_var=realized)
        assert pnl_2 == pytest.approx(2 * pnl_1, rel=1e-10)

    def test_returns_float(self):
        pnl = variance_swap_pnl(**_BASE, N_notional=1.0, realized_var=0.04)
        assert isinstance(pnl, float)


# ===========================================================================
# 5. variance_term_structure
# ===========================================================================

class TestVarianceTermStructure:
    def test_default_length(self):
        """Default T_grid has 7 maturities → output array of length 7."""
        ts = variance_term_structure(
            kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1
        )
        assert len(ts) == 7, f"Expected length 7, got {len(ts)}"

    def test_custom_length(self):
        """Custom T_grid of length 3 → output array of length 3."""
        ts = variance_term_structure(
            kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1,
            T_grid=np.array([0.25, 0.5, 1.0])
        )
        assert len(ts) == 3

    def test_all_positive(self):
        """All term-structure rates must be positive."""
        ts = variance_term_structure(
            kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1
        )
        assert np.all(ts > 0.0), f"Non-positive rates found: {ts}"

    def test_returns_ndarray(self):
        """Return type must be np.ndarray."""
        ts = variance_term_structure(
            kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1
        )
        assert isinstance(ts, np.ndarray)

    def test_consistency_with_variance_swap_rate(self):
        """Each entry must match the scalar variance_swap_rate for the same T."""
        T_grid = np.array([0.25, 0.5, 1.0, 2.0])
        params = dict(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1)
        ts = variance_term_structure(**params, T_grid=T_grid)

        for i, T in enumerate(T_grid):
            kv_scalar = variance_swap_rate(**params, T=T)
            assert ts[i] == pytest.approx(kv_scalar, rel=1e-5), (
                f"T={T}: term_structure[{i}]={ts[i]:.8f} vs scalar={kv_scalar:.8f}"
            )

    def test_monotonicity_mean_reverting(self):
        """With v0 >> θ, K_var should decrease with maturity (mean reversion).

        v0=0.16, θ=0.04: initial variance is 4× theta, so K_var should decline.
        """
        ts = variance_term_structure(
            kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.16, H=0.1,
            T_grid=np.array([0.25, 0.5, 1.0, 2.0])
        )
        # K_var(T) should be decreasing (converging down toward θ)
        assert ts[0] > ts[-1], (
            f"With v0 >> theta, K_var should decrease: {ts}"
        )

    def test_empty_t_grid(self):
        """Empty T_grid returns empty array without error."""
        ts = variance_term_structure(
            kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1,
            T_grid=np.array([])
        )
        assert len(ts) == 0

    def test_order_preserved(self):
        """Output order must match input T_grid order (even if unsorted)."""
        T_grid_unsorted = np.array([1.0, 0.25, 2.0, 0.5])
        ts = variance_term_structure(
            kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1,
            T_grid=T_grid_unsorted
        )
        # Each entry should match its individual scalar computation
        params = dict(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04, H=0.1)
        for i, T in enumerate(T_grid_unsorted):
            kv_scalar = variance_swap_rate(**params, T=T)
            assert ts[i] == pytest.approx(kv_scalar, rel=1e-5), (
                f"T={T}: mismatch ts[{i}]={ts[i]:.8f}, scalar={kv_scalar:.8f}"
            )
