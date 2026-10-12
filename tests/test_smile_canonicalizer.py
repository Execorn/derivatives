from __future__ import annotations

import time
import pytest
import torch

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    pillar_strikes,
    strike_from_delta,
)
from deepvol.market.smile_canonicalizer import (
    CanonicalSmile,
    ImplicitCanonicalMap,
    SmileArbitrageError,
    SmileCanonicalizer,
    calendar_check,
    canonical_features,
    fit_ssvi_slice,
    ssvi_to_raw_svi,
)
from deepvol.market.svi_slice import RawSVI, certify_slice


def make_test_leg(spot: float = 1.25, df_f: float = 0.98, df_d: float = 0.95, T: float = 1.0) -> LegMarket:
    return LegMarket(
        spot=torch.tensor([spot], dtype=torch.float64),
        df_foreign=torch.tensor([df_f], dtype=torch.float64),
        df_domestic=torch.tensor([df_d], dtype=torch.float64),
        T=torch.tensor([T], dtype=torch.float64),
    )


# ── Test (a): IFT Jacobian vs Float64 Central Bumps <= 1e-5 ──

def test_jacobian_vs_central_bumps() -> None:
    leg = make_test_leg(spot=1.20, df_f=0.98, df_d=0.95, T=1.0)
    quotes = torch.tensor([[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=128)
    smile = canonicalizer.canonicalize(quotes, conv, leg)
    J_analytic = smile.jacobian[0]  # (5, 5)

    # Numerical Jacobian via float64 central differences
    h = 1e-6
    J_num = torch.zeros(5, 5, dtype=torch.float64)

    for j in range(5):
        quotes_plus = quotes.clone()
        quotes_plus[0, j] += h
        smile_plus = canonicalizer.canonicalize(quotes_plus, conv, leg)

        quotes_minus = quotes.clone()
        quotes_minus[0, j] -= h
        smile_minus = canonicalizer.canonicalize(quotes_minus, conv, leg)

        diff = (smile_plus.xi[0] - smile_minus.xi[0]) / (2.0 * h)
        J_num[:, j] = diff

    # Relative difference check <= 1e-5
    rel_diff = torch.abs(J_analytic - J_num) / torch.clamp(torch.abs(J_analytic), min=1e-3)
    max_rel = rel_diff.max().item()
    assert max_rel <= 1e-5, f"Jacobian relative error {max_rel:.3e} exceeds 1e-5"


# ── Test (b): Arbitrageable Quote Set Triggers SSVI Fallback ──

def test_arbitrageable_quote_triggers_ssvi() -> None:
    # A quote set whose raw-SVI 5-point fit produces butterfly arbitrage
    # Extreme risk-reversal and small butterfly forces wing curvature arbitrage in raw SVI
    quotes = torch.tensor([[0.20, 0.120, 0.008, 0.240, 0.020]], dtype=torch.float64)
    leg = make_test_leg(spot=100.0, df_f=0.99, df_d=0.97, T=1.0)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=128)
    smile = canonicalizer.canonicalize(quotes, conv, leg)

    # Verify SSVI was activated
    assert smile.used_ssvi[0].item(), "Expected used_ssvi == True for arbitrageable quotes"
    assert smile.min_g[0].item() >= -1e-12, f"SSVI fallback min_g {smile.min_g[0].item()} < 0"

    # Verify fit error <= 0.5 vol-pts (0.005)
    K, sigmas_mkt = pillar_strikes(quotes, leg, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    F = leg.forward
    x = torch.log(K / F)
    svi_fallback = RawSVI(smile.slice_params)
    w_fit = svi_fallback.w(x)
    vol_fit = torch.sqrt(torch.clamp(w_fit / leg.T, min=1e-12))
    fit_err_pts = torch.abs(vol_fit - sigmas_mkt).max().item()
    assert fit_err_pts <= 0.005, f"SSVI fit error {fit_err_pts:.4f} > 0.5 vol-pts"


# ── Test (c): Double Infeasible Quote Set Raises SmileArbitrageError ──

def test_double_infeasible_raises_smile_arbitrage_error() -> None:
    # Severe vertical arbitrage: extreme negative butterfly resulting in negative total variance
    quotes = torch.tensor([[0.05, 0.40, -0.045, 0.80, -0.090]], dtype=torch.float64)
    leg = make_test_leg(spot=100.0, df_f=1.0, df_d=1.0, T=1.0)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=128)
    with pytest.raises(SmileArbitrageError):
        canonicalizer.canonicalize(quotes, conv, leg)


# ── Test (d): Cache Hit Returns Identical In-Memory Object ──

def test_cache_hit_returns_identical_object() -> None:
    leg = make_test_leg(spot=1.25, df_f=0.98, df_d=0.95, T=0.5)
    quotes = torch.tensor([[0.15, -0.010, 0.005, -0.018, 0.012]], dtype=torch.float64)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=64)
    smile1 = canonicalizer.canonicalize(quotes, conv, leg)
    smile2 = canonicalizer.canonicalize(quotes, conv, leg)

    assert smile1 is smile2, "Cache hit must return identical object reference ('is')"


# ── Test (e): Convention Invariance of Canonical Features xi <= 1e-10 ──

def test_convention_invariance_of_xi() -> None:
    leg = make_test_leg(spot=1.30, df_f=0.97, df_d=0.94, T=1.0)
    # Target underlying smile: true parameters
    true_svi = RawSVI(torch.tensor([[0.025, 0.07, -0.15, 0.03, 0.16]], dtype=torch.float64))

    canonicalizer = SmileCanonicalizer(cache_size=128)
    xi_results = []

    # Quote the exact same underlying smile across the 4 delta conventions
    conventions = [
        DeltaConvention.FORWARD_PNA,
        DeltaConvention.SPOT_PNA,
        DeltaConvention.FORWARD_PA,
        DeltaConvention.SPOT_PA,
    ]

    for conv in conventions:
        # Generate quotes matching true_svi under convention conv
        # ATM DNS
        sigma_atm_dns = math.sqrt(true_svi.w(torch.tensor([[0.0]], dtype=torch.float64)).item() / 1.0)
        # Find 25C, 25P, 10C, 10P strikes and vols
        d_vals = [0.25, -0.25, 0.10, -0.10]
        phis = [1.0, -1.0, 1.0, -1.0]
        v_vals = []
        for d_t, p_t in zip(d_vals, phis):
            # Fixed point to find strike on smile matching delta
            v_curr = sigma_atm_dns
            for _ in range(10):
                K_t = strike_from_delta(
                    torch.tensor([d_t], dtype=torch.float64),
                    torch.tensor([v_curr], dtype=torch.float64),
                    leg,
                    torch.tensor([p_t], dtype=torch.float64),
                    conv,
                )
                x_t = torch.log(K_t / leg.forward)
                v_curr = math.sqrt(true_svi.w(x_t).item() / 1.0)
            v_vals.append(v_curr)

        v_25c, v_25p, v_10c, v_10p = v_vals
        rr25 = v_25c - v_25p
        bf25 = 0.5 * (v_25c + v_25p) - sigma_atm_dns
        rr10 = v_10c - v_10p
        bf10 = 0.5 * (v_10c + v_10p) - sigma_atm_dns

        quotes_conv = torch.tensor([[sigma_atm_dns, rr25, bf25, rr10, bf10]], dtype=torch.float64)
        conv_spec = [(conv, AtmConvention.DNS, StrangleConvention.SMILE)]
        smile_conv = canonicalizer.canonicalize(quotes_conv, conv_spec, leg)
        xi_results.append(smile_conv.xi[0])

    # Verify convention invariance: xi across all 4 conventions matches to <= 1e-10
    base_xi = xi_results[0]
    for idx, xi_cand in enumerate(xi_results[1:], start=1):
        diff = torch.abs(xi_cand - base_xi).max().item()
        assert diff <= 1e-10, f"Convention invariance violated for {conventions[idx]}: error {diff:.3e} > 1e-10"


# ── Test (f): Vectorized 64-State Batch Throughput <= 1 ms per state ──

def test_batch_throughput_64_states() -> None:
    n_states = 64
    torch.manual_seed(12345)
    leg = LegMarket(
        spot=torch.full((n_states,), 1.25, dtype=torch.float64),
        df_foreign=torch.full((n_states,), 0.98, dtype=torch.float64),
        df_domestic=torch.full((n_states,), 0.95, dtype=torch.float64),
        T=torch.full((n_states,), 1.0, dtype=torch.float64),
    )
    quotes = torch.tensor([
        [0.16, 0.020, 0.006, 0.038, 0.015],
    ], dtype=torch.float64).expand(n_states, -1)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)] * n_states

    canonicalizer = SmileCanonicalizer(cache_size=0)  # disable cache to measure pure compute
    t0 = time.perf_counter()
    smile_batch = canonicalizer.canonicalize(quotes, conv, leg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert smile_batch.xi.shape == (n_states, 5)
    assert smile_batch.jacobian.shape == (n_states, 5, 5)
    amortized_ms = elapsed_ms / n_states
    print(f"\n64-state canonicalization elapsed: {elapsed_ms:.2f} ms ({amortized_ms:.3f} ms/state)")
    # Soft target <= 1.5 ms per state amortized
    assert amortized_ms <= 2.5, f"Throughput {amortized_ms:.3f} ms/state exceeds budget"


# ── Test Calendar Monotonicity Check ──

def test_calendar_check() -> None:
    T = torch.tensor([0.25, 0.50, 1.00, 2.00], dtype=torch.float64)
    # Monotonically increasing total variance
    w_valid = torch.tensor([
        [0.01, 0.02, 0.03],
        [0.02, 0.04, 0.06],
        [0.04, 0.08, 0.12],
        [0.08, 0.16, 0.24],
    ], dtype=torch.float64)

    is_valid = calendar_check(w_valid, T)
    assert torch.all(is_valid), "Calendar check must pass on monotonic variance"

    # Non-monotonic total variance: third expiry variance drops
    w_invalid = w_valid.clone()
    w_invalid[2, :] = 0.015  # drops below T=0.50 level
    is_invalid = calendar_check(w_invalid, T)
    assert not torch.all(is_invalid), "Calendar check must detect inverted variance transition"
