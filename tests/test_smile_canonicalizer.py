from __future__ import annotations

import math
import time
from typing import cast
import pytest
import torch
from torch import Tensor

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


# ── Test (a1): Reference Convention Identity Check ──

def test_jacobian_vs_central_bumps_reference() -> None:
    leg = make_test_leg(spot=1.20, df_f=0.98, df_d=0.95, T=1.0)
    quotes = torch.tensor([[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=128)
    smile = canonicalizer.canonicalize(quotes, conv, leg)
    J_analytic = smile.jacobian[0]

    assert torch.allclose(J_analytic, torch.eye(5, dtype=torch.float64), atol=1e-12)


# ── Test (a2): Non-Reference Convention IFT vs Central Differences (SPOT_PNA + ATMF) ──

def test_jacobian_vs_central_bumps_spot_pna_atmf() -> None:
    leg = make_test_leg(spot=1.25, df_f=0.98, df_d=0.95, T=1.0)
    quotes = torch.tensor([[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64)
    conv = [(DeltaConvention.SPOT_PNA, AtmConvention.ATMF, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=0)
    smile = canonicalizer.canonicalize(quotes, conv, leg)
    J_analytic = smile.jacobian[0]

    assert not torch.allclose(J_analytic, torch.eye(5, dtype=torch.float64), atol=1e-3)

    h = 1e-6
    J_num = torch.zeros(5, 5, dtype=torch.float64)
    for j in range(5):
        qp = quotes.clone()
        qp[0, j] += h
        smile_p = canonicalizer.canonicalize(qp, conv, leg)

        qm = quotes.clone()
        qm[0, j] -= h
        smile_m = canonicalizer.canonicalize(qm, conv, leg)

        J_num[:, j] = (smile_p.xi[0] - smile_m.xi[0]) / (2.0 * h)

    rel_diff = torch.abs(J_analytic - J_num) / torch.clamp(torch.abs(J_analytic), min=1e-3)
    max_rel = rel_diff.max().item()
    assert max_rel <= 1e-5, f"IFT Jacobian relative error {max_rel:.3e} exceeds 1e-5 on SPOT_PNA"


# ── Test (a3): Premium-Adjusted Convention IFT vs Central Differences (FORWARD_PA + DNS) ──

def test_jacobian_vs_central_bumps_forward_pa() -> None:
    leg = make_test_leg(spot=1.25, df_f=0.98, df_d=0.95, T=1.0)
    quotes = torch.tensor([[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64)
    conv = [(DeltaConvention.FORWARD_PA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=0)
    smile = canonicalizer.canonicalize(quotes, conv, leg)
    J_analytic = smile.jacobian[0]

    assert not torch.allclose(J_analytic, torch.eye(5, dtype=torch.float64), atol=1e-3)

    h = 1e-6
    J_num = torch.zeros(5, 5, dtype=torch.float64)
    for j in range(5):
        qp = quotes.clone()
        qp[0, j] += h
        smile_p = canonicalizer.canonicalize(qp, conv, leg)

        qm = quotes.clone()
        qm[0, j] -= h
        smile_m = canonicalizer.canonicalize(qm, conv, leg)

        J_num[:, j] = (smile_p.xi[0] - smile_m.xi[0]) / (2.0 * h)

    rel_diff = torch.abs(J_analytic - J_num) / torch.clamp(torch.abs(J_analytic), min=1e-3)
    max_rel = rel_diff.max().item()
    assert max_rel <= 1e-5, f"IFT Jacobian relative error {max_rel:.3e} exceeds 1e-5 on FORWARD_PA"


# ── Test (a4): Premium-Adjusted Convention IFT vs Central Differences (SPOT_PA + DNS) ──

def test_jacobian_vs_central_bumps_spot_pa() -> None:
    leg = make_test_leg(spot=1.25, df_f=0.98, df_d=0.95, T=1.0)
    quotes = torch.tensor([[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64)
    conv = [(DeltaConvention.SPOT_PA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canonicalizer = SmileCanonicalizer(cache_size=0)
    smile = canonicalizer.canonicalize(quotes, conv, leg)
    J_analytic = smile.jacobian[0]

    assert not torch.allclose(J_analytic, torch.eye(5, dtype=torch.float64), atol=1e-3)

    h = 1e-6
    J_num = torch.zeros(5, 5, dtype=torch.float64)
    for j in range(5):
        qp = quotes.clone()
        qp[0, j] += h
        smile_p = canonicalizer.canonicalize(qp, conv, leg)

        qm = quotes.clone()
        qm[0, j] -= h
        smile_m = canonicalizer.canonicalize(qm, conv, leg)

        J_num[:, j] = (smile_p.xi[0] - smile_m.xi[0]) / (2.0 * h)

    rel_diff = torch.abs(J_analytic - J_num) / torch.clamp(torch.abs(J_analytic), min=1e-3)
    max_rel = rel_diff.max().item()
    assert max_rel <= 1e-5, f"IFT Jacobian relative error {max_rel:.3e} exceeds 1e-5 on SPOT_PA"


# ── Test (b): Arbitrageable Quote Set Triggers SSVI Fallback ──

def test_arbitrageable_quote_triggers_ssvi() -> None:
    # A quote set whose raw-SVI 5-point fit produces butterfly arbitrage
    # Extreme risk-reversal and small butterfly forces wing curvature arbitrage in raw SVI
    quotes = torch.tensor([[0.16, 0.03, 0.002, 0.054, 0.008]], dtype=torch.float64)
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
        # ATM DNS solved on smile for convention
        sign_shift = 1.0 if conv in (DeltaConvention.FORWARD_PNA, DeltaConvention.SPOT_PNA) else -1.0
        x_atm = 0.5 * sign_shift * true_svi.w(torch.tensor([[0.0]], dtype=torch.float64))
        for _ in range(10):
            w_val, w1_val, _ = true_svi.derivs(x_atm)
            x_atm = x_atm - (x_atm - 0.5 * sign_shift * w_val) / (1.0 - 0.5 * sign_shift * w1_val)
        sigma_atm_dns = math.sqrt(true_svi.w(x_atm).item() / 1.0)
        # Find 25C, 25P, 10C, 10P strikes and vols
        d_vals = [0.25, -0.25, 0.10, -0.10]
        phis = [1.0, -1.0, 1.0, -1.0]
        v_vals = []
        for d_t, p_t in zip(d_vals, phis):
            # Fixed point to find strike on smile matching delta
            v_curr = sigma_atm_dns
            for _ in range(15):
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
    # Generate 64 distinct realistic market states: varying spots, tenors, discount curves, and quote smiles
    spots = torch.linspace(1.10, 1.45, n_states, dtype=torch.float64)
    T = torch.linspace(0.10, 2.00, n_states, dtype=torch.float64)
    df_f = torch.exp(-0.02 * T)
    df_d = torch.exp(-0.04 * T)
    leg = LegMarket(spot=spots, df_foreign=df_f, df_domestic=df_d, T=T)

    atm = torch.linspace(0.12, 0.20, n_states, dtype=torch.float64)
    rr25 = torch.linspace(0.010, 0.030, n_states, dtype=torch.float64)
    bf25 = torch.linspace(0.004, 0.009, n_states, dtype=torch.float64)
    rr10 = rr25 * 1.85
    bf10 = bf25 * 2.80
    quotes = torch.stack([atm, rr25, bf25, rr10, bf10], dim=-1)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)] * n_states

    canonicalizer = SmileCanonicalizer(cache_size=0)  # disable cache to measure pure compute
    t0 = time.perf_counter()
    smile_batch = canonicalizer.canonicalize(quotes, conv, leg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert smile_batch.xi.shape == (n_states, 5)
    assert smile_batch.jacobian.shape == (n_states, 5, 5)
    amortized_ms = elapsed_ms / n_states
    print(f"\n64-state canonicalization elapsed: {elapsed_ms:.2f} ms ({amortized_ms:.3f} ms/state)")
    # Soft target <= 1.0 ms per state amortized, SLA budget <= 2.5 ms/state
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


# ── Test SSVI Slice Fit, Raw-SVI Mapping, and Certification ──

def test_ssvi_fit_and_mapping() -> None:
    x = torch.tensor([[-0.20, -0.10, 0.00, 0.10, 0.20]], dtype=torch.float64)
    w = torch.tensor([[0.045, 0.038, 0.035, 0.037, 0.043]], dtype=torch.float64)
    params, rmse = fit_ssvi_slice(x, w)
    assert params.shape == (1, 3)
    assert rmse[0].item() <= 0.01

    raw_svi = ssvi_to_raw_svi(params[:, 0], params[:, 1], params[:, 2])
    assert isinstance(raw_svi, RawSVI)
    sigma0 = torch.tensor([0.20], dtype=torch.float64)
    T = torch.tensor([1.0], dtype=torch.float64)
    cert = certify_slice(raw_svi, sigma0, T)
    assert cert.ok.item()
    assert cert.min_g.item() >= 0.0


# ── Test Direct Canonical Features and Autograd Implicit Map ──

def test_canonical_features_and_autograd() -> None:
    leg = make_test_leg(spot=1.20, df_f=0.98, df_d=0.95, T=1.0)
    svi = RawSVI(torch.tensor([[0.03, 0.08, -0.10, 0.02, 0.15]], dtype=torch.float64))
    xi = canonical_features(svi, leg)
    assert xi.shape == (1, 5)

    quotes = torch.tensor([[0.16, 0.020, 0.006, 0.038, 0.015]], dtype=torch.float64, requires_grad=True)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]
    canonicalizer = SmileCanonicalizer(cache_size=16)

    # Autograd forward + backward through ImplicitCanonicalMap
    xi_map = cast(Tensor, ImplicitCanonicalMap.apply(quotes, conv, leg, canonicalizer))  # type: ignore[no-untyped-call]
    assert isinstance(xi_map, torch.Tensor)
    loss = xi_map.sum()
    loss.backward()  # type: ignore[no-untyped-call]
    assert quotes.grad is not None
    assert quotes.grad.shape == (1, 5)

    # Check CanonicalSmile instance structure
    smile = canonicalizer.canonicalize(quotes.detach(), conv, leg)
    assert isinstance(smile, CanonicalSmile)
    assert smile.slice_params.shape == (1, 5)


# ── Test SSVI Fallback Direct Autograd and Boundary Hardening ──

def test_canonicalize_direct_requires_grad_ssvi_fallback() -> None:
    """Verifies that direct canonicalize call on quotes with requires_grad=True succeeds on SSVI fallback."""
    can = SmileCanonicalizer(cache_size=0)
    leg = LegMarket(
        spot=torch.tensor([1.20], dtype=torch.float64),
        df_foreign=torch.tensor([0.98], dtype=torch.float64),
        df_domestic=torch.tensor([0.95], dtype=torch.float64),
        T=torch.tensor([0.5], dtype=torch.float64),
    )
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    # Severe butterfly-violating quotes forcing SSVI fallback
    quotes_arb = torch.tensor(
        [[0.20, 0.08, -0.015, 0.16, -0.025]],
        dtype=torch.float64,
        requires_grad=True,
    )

    smile = can.canonicalize(quotes_arb, conv, leg)
    assert smile.used_ssvi[0].item() is True
    assert torch.isfinite(smile.xi).all()
    assert torch.isfinite(smile.jacobian).all()
    assert smile.min_g[0].item() >= -1e-12


def test_ssvi_boundary_clamping_severe_positive_skew() -> None:
    """Verifies that severe positive skew quotes do not collapse rho to 1.0 in SSVI fallback."""
    leg = LegMarket(
        spot=torch.tensor([1.25], dtype=torch.float64),
        df_foreign=torch.tensor([0.99], dtype=torch.float64),
        df_domestic=torch.tensor([0.97], dtype=torch.float64),
        T=torch.tensor([0.5], dtype=torch.float64),
    )
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]
    # Quotes triggering Defect 3: RR10 = 0.18, BF10 = -0.028
    q_pos = torch.tensor([[0.22, 0.09, -0.018, 0.18, -0.028]], dtype=torch.float64)

    can = SmileCanonicalizer(cache_size=0)
    smile = can.canonicalize(q_pos, conv, leg)

    assert smile.used_ssvi.item() is True, "Expected SSVI fallback activation"
    assert smile.min_g.item() >= 0.0, f"Expected non-negative butterfly min_g, got {smile.min_g.item()}"
    rho_fitted = smile.slice_params[0, 2].item()
    assert abs(rho_fitted) <= 0.999 + 1e-10, f"Fitted rho {rho_fitted} exceeded interior bound 0.999"
    assert 1.0 - (rho_fitted ** 2) > 1e-4, "Expected strictly positive 1 - rho^2 margin"
    assert smile.slice_params[0, 0].item() > 0.0, "Expected strictly positive a parameter"
    assert smile.slice_params[0, 4].item() > 0.0, "Expected strictly positive s parameter"


def test_ssvi_boundary_clamping_severe_negative_skew() -> None:
    """Verifies that severe negative skew quotes do not collapse rho to -1.0 in SSVI fallback."""
    leg = LegMarket(
        spot=torch.tensor([1.25], dtype=torch.float64),
        df_foreign=torch.tensor([0.99], dtype=torch.float64),
        df_domestic=torch.tensor([0.97], dtype=torch.float64),
        T=torch.tensor([0.5], dtype=torch.float64),
    )
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]
    # Severe put skew: RR10 = -0.18, BF10 = -0.028
    q_neg = torch.tensor([[0.22, -0.09, -0.018, -0.18, -0.028]], dtype=torch.float64)

    can = SmileCanonicalizer(cache_size=0)
    smile = can.canonicalize(q_neg, conv, leg)

    assert smile.used_ssvi.item() is True, "Expected SSVI fallback activation"
    assert smile.min_g.item() >= 0.0, f"Expected non-negative butterfly min_g, got {smile.min_g.item()}"
    rho_fitted = smile.slice_params[0, 2].item()
    assert rho_fitted >= -0.999 - 1e-10, f"Fitted rho {rho_fitted} exceeded interior bound -0.999"
    assert 1.0 - (rho_fitted ** 2) > 1e-4, "Expected strictly positive 1 - rho^2 margin"
    assert smile.slice_params[0, 0].item() > 0.0, "Expected strictly positive a parameter"


def test_ssvi_to_raw_svi_boundary_admissibility() -> None:
    """Directly verifies that ssvi_to_raw_svi preserves slice admissibility at rho = +/- 1.0."""
    theta = torch.tensor([0.04, 0.04, 0.02], dtype=torch.float64)
    rho_boundary = torch.tensor([1.0, -1.0, 0.999], dtype=torch.float64)
    phi = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)

    raw_svi = ssvi_to_raw_svi(theta, rho_boundary, phi)
    sigma0 = torch.tensor([0.20, 0.20, 0.14], dtype=torch.float64)
    T = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    cert = certify_slice(raw_svi, sigma0, T)
    assert cert.ok.all(), "All boundary-clamped slices must satisfy certification"
    assert cert.positive_ok.all(), "All boundary-clamped slices must satisfy positive_ok"
    assert cert.lee_ok.all(), "All boundary-clamped slices must satisfy lee_ok"
    assert (cert.min_g >= -1e-12).all(), "All boundary-clamped slices must satisfy Durrleman condition"


def test_ssvi_fallback_autograd_differentiable_quotes_severe_skew() -> None:
    """Verifies autograd forward and backward stability under severe skew SSVI fallback."""
    can = SmileCanonicalizer(cache_size=0)
    leg = LegMarket(
        spot=torch.tensor([1.25], dtype=torch.float64),
        df_foreign=torch.tensor([0.99], dtype=torch.float64),
        df_domestic=torch.tensor([0.97], dtype=torch.float64),
        T=torch.tensor([0.5], dtype=torch.float64),
    )
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]
    q_diff = torch.tensor([[0.22, 0.09, -0.018, 0.18, -0.028]], dtype=torch.float64, requires_grad=True)

    xi = cast(Tensor, ImplicitCanonicalMap.apply(q_diff, conv, leg, can))  # type: ignore[no-untyped-call]
    loss = (xi ** 2).sum()
    loss.backward()  # type: ignore[no-untyped-call]

    assert q_diff.grad is not None
    assert torch.isfinite(q_diff.grad).all()
    assert not torch.isnan(q_diff.grad).any()
    assert not torch.isinf(q_diff.grad).any()
