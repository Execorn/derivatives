from __future__ import annotations

import math
from typing import cast
import numpy as np
import pytest
import torch
from torch import Tensor

from deepvol.market.fx_conventions import (
    AtmConvention,
    ConventionSolveError,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    fx_delta,
    market_to_smile_strangle,
    pa_call_delta_argmax,
    strike_from_delta,
)
from deepvol.market.smile_canonicalizer import (
    ImplicitCanonicalMap,
    SmileArbitrageError,
    SmileCanonicalizer,
    calendar_check,
)


def _make_leg(spot: float, df_f: float, df_d: float, T: float) -> LegMarket:
    return LegMarket(
        spot=torch.tensor([spot], dtype=torch.float64),
        df_foreign=torch.tensor([df_f], dtype=torch.float64),
        df_domestic=torch.tensor([df_d], dtype=torch.float64),
        T=torch.tensor([T], dtype=torch.float64),
    )


# ── Challenge 1: Extreme Cartesian Stress Matrix on PNA Conventions ──

def test_stress_cartesian_pna_roundtrip() -> None:
    """Stress tests PNA delta inversion across extreme spots, tenors, vols, and strikes."""
    np.seterr(all="raise")
    torch.autograd.set_detect_anomaly(True)

    spots = [1e-5, 1.0, 1e5]
    tenors = [1/365, 0.1, 1.0, 5.0, 10.0]
    vols = [0.01, 0.10, 0.50, 1.5, 3.0]
    deltas = [0.05, 0.10, 0.25, 0.45]

    for s in spots:
        for t in tenors:
            for v in vols:
                leg = _make_leg(s, 0.98, 0.95, t)
                sig = torch.tensor([v], dtype=torch.float64)

                for conv in [DeltaConvention.FORWARD_PNA, DeltaConvention.SPOT_PNA]:
                    for phi_val in [1.0, -1.0]:
                        phi = torch.tensor([phi_val], dtype=torch.float64)
                        for d_val in deltas:
                            target = torch.tensor([d_val * phi_val], dtype=torch.float64)
                            K = strike_from_delta(target, sig, leg, phi, conv)
                            assert torch.isfinite(K).all() and (K > 0).all()

                            rec = fx_delta(K, sig, leg, phi, conv)
                            assert torch.isfinite(rec).all()
                            err = torch.abs(rec - target).item()
                            assert err <= 1e-12, (
                                f"PNA residual {err:.3e} exceeds 1e-12 at s={s}, t={t}, v={v}, d={d_val}, conv={conv}"
                            )


# ── Challenge 2: PA Argmax and Boundary Inversion ──

def test_stress_pa_argmax_and_inversion() -> None:
    """Stress tests Premium-Adjusted conventions, argmax stationary condition, and bounds."""
    np.seterr(all="raise")

    vols = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    tenors = [1/365, 0.1, 1.0, 3.0]

    for v in vols:
        for t in tenors:
            leg = _make_leg(1.25, 0.98, 0.95, t)
            sig = torch.tensor([v], dtype=torch.float64)

            # Argmax strike K_max must be positive, finite, and strictly non-zero
            K_max = pa_call_delta_argmax(sig, leg)
            assert torch.isfinite(K_max).all() and (K_max > 0).all()

            # Verify argmax derivative stationary condition: tau * N(d2) - n(d2) == 0
            tau = v * math.sqrt(t)
            d2_max = (math.log(leg.forward.item() / K_max.item()) - 0.5 * tau ** 2) / tau
            N2_val = 0.5 * (1.0 + math.erf(d2_max / math.sqrt(2.0)))
            n2_val = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d2_max ** 2)
            stat_res = abs(tau * N2_val - n2_val)
            assert stat_res <= 1e-11, f"Argmax stationary residual {stat_res:.3e} > 1e-11 at v={v}, t={t}"

            d_max = fx_delta(K_max, sig, leg, torch.tensor([1.0], dtype=torch.float64), DeltaConvention.FORWARD_PA).item()

            # PA Call: Invert feasible delta targets (below d_max)
            for d_target in [0.05, 0.10, 0.25]:
                if d_target < d_max - 1e-4:
                    target_ten = torch.tensor([d_target], dtype=torch.float64)
                    phi_call = torch.tensor([1.0], dtype=torch.float64)
                    K_call = strike_from_delta(target_ten, sig, leg, phi_call, DeltaConvention.FORWARD_PA)
                    assert torch.isfinite(K_call).all() and (K_call > 0).all()
                    rec_call = fx_delta(K_call, sig, leg, phi_call, DeltaConvention.FORWARD_PA)
                    assert torch.abs(rec_call - target_ten).item() <= 1e-12

            # PA Put: Globally monotonic, all deltas in [0.05, 0.45] must invert
            for d_target in [0.05, 0.10, 0.25, 0.45]:
                target_ten = torch.tensor([-d_target], dtype=torch.float64)
                phi_put = torch.tensor([-1.0], dtype=torch.float64)
                K_put = strike_from_delta(target_ten, sig, leg, phi_put, DeltaConvention.FORWARD_PA)
                assert torch.isfinite(K_put).all() and (K_put > 0).all()
                rec_put = fx_delta(K_put, sig, leg, phi_put, DeltaConvention.FORWARD_PA)
                assert torch.abs(rec_put - target_ten).item() <= 1e-12

            # Unfeasible PA Call delta (> d_max) must raise ConventionSolveError cleanly
            unfeasible_delta = d_max + 0.10
            if unfeasible_delta < 1.0:
                with pytest.raises(ConventionSolveError):
                    strike_from_delta(
                        torch.tensor([unfeasible_delta], dtype=torch.float64),
                        sig,
                        leg,
                        torch.tensor([1.0], dtype=torch.float64),
                        DeltaConvention.FORWARD_PA,
                    )


# ── Challenge 3: Extreme Tail Boundary Trapping & Floating Point Defense ──

def test_stress_extreme_tail_boundary_trapping() -> None:
    """Validates that extreme strikes and deep tails produce zero NaNs or Infs."""
    np.seterr(all="raise")
    torch.autograd.set_detect_anomaly(True)

    leg = _make_leg(100.0, 0.98, 0.95, 1.0)
    sig = torch.tensor([0.20], dtype=torch.float64)

    # Extreme strike ratios K/F from 1e-6 (deep ITM put/OTM call) to 1e6 (deep OTM call/ITM put)
    extreme_strikes = leg.forward * torch.tensor([1e-6, 1e-4, 0.01, 1.0, 100.0, 1e4, 1e6], dtype=torch.float64)

    for K in extreme_strikes:
        K_ten = K.unsqueeze(0)
        for conv in DeltaConvention:
            for phi_val in [1.0, -1.0]:
                phi = torch.tensor([phi_val], dtype=torch.float64)
                d_val = fx_delta(K_ten, sig, leg, phi, conv)
                assert torch.isfinite(d_val).all()
                assert not torch.isnan(d_val).any()
                assert not torch.isinf(d_val).any()
                if conv in (DeltaConvention.FORWARD_PNA, DeltaConvention.SPOT_PNA):
                    assert (d_val.abs() >= 0.0).all() and (d_val.abs() <= 1.0 + 1e-12).all()


# ── Challenge 4: SVI / SSVI Robustness under Pathological & Noisy Quotes ──

def test_stress_svi_noisy_monte_carlo() -> None:
    """Stress tests SVI slice fitting and SSVI fallback across 100 randomized noisy quote sets."""
    np.seterr(all="raise")

    rng = np.random.default_rng(seed=42)
    can = SmileCanonicalizer(cache_size=256)
    leg = _make_leg(1.15, 0.99, 0.97, 0.5)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    raw_svi_count = 0
    ssvi_fallback_count = 0
    arbitrage_error_count = 0

    for _ in range(100):
        atm = float(rng.uniform(0.08, 0.35))
        # Random risk reversals and butterflies with occasional arbitrage-inducing noise
        rr25 = float(rng.uniform(-0.08, 0.08))
        bf25 = float(rng.uniform(-0.005, 0.030))
        rr10 = float(rng.uniform(-0.15, 0.15))
        bf10 = float(rng.uniform(-0.010, 0.060))

        # Enforce minimum positive wing vols >= 0.01
        if (atm + bf25 - 0.5 * abs(rr25) < 0.01) or (atm + bf10 - 0.5 * abs(rr10) < 0.01):
            continue

        quotes = torch.tensor([[atm, rr25, bf25, rr10, bf10]], dtype=torch.float64)

        try:
            canon = can.canonicalize(quotes, conv, leg)
            assert torch.isfinite(canon.slice_params).all()
            assert torch.isfinite(canon.xi).all()
            assert torch.isfinite(canon.jacobian).all()
            assert canon.min_g.item() >= -1e-12, f"Butterfly violation min_g {canon.min_g.item()} < 0"

            if canon.used_ssvi.item():
                ssvi_fallback_count += 1
            else:
                raw_svi_count += 1
        except SmileArbitrageError:
            arbitrage_error_count += 1

    # Ensure all scenarios resolved to either clean SVI, valid SSVI fallback, or caught SmileArbitrageError
    total_processed = raw_svi_count + ssvi_fallback_count + arbitrage_error_count
    assert total_processed >= 80, f"Expected at least 80 valid test scenarios, got {total_processed}"
    assert raw_svi_count > 0, "Expected at least one raw SVI fit"
    assert ssvi_fallback_count > 0, "Expected at least one SSVI fallback activation"


# ── Challenge 5: Autograd Backward & Anomaly Detection ──

def test_stress_autograd_jacobian_anomaly() -> None:
    """Stress tests PyTorch autograd anomaly detection through ImplicitCanonicalMap."""
    torch.autograd.set_detect_anomaly(True)
    np.seterr(all="raise")

    can = SmileCanonicalizer(cache_size=64)
    leg = _make_leg(1.08, 0.99, 0.96, 1.0)
    conv = [(DeltaConvention.SPOT_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    # Realistic EUR/USD quote set
    quotes = torch.tensor([[0.12, -0.012, 0.004, -0.025, 0.010]], dtype=torch.float64, requires_grad=True)

    xi = cast(Tensor, ImplicitCanonicalMap.apply(quotes, conv, leg, can))  # type: ignore[no-untyped-call]
    assert xi.shape == (1, 5)

    # Backpropagate through arbitrary scalar loss
    loss = (xi ** 2).sum()
    loss.backward()  # type: ignore[no-untyped-call]

    assert quotes.grad is not None
    assert torch.isfinite(quotes.grad).all()
    assert not torch.isnan(quotes.grad).any()
    assert not torch.isinf(quotes.grad).any()


# ── Challenge 6: Market Strangle Inversion Precision ──

def test_stress_market_strangle_inversion() -> None:
    """Stress tests broker market strangle inversion under tight tolerance."""
    np.seterr(all="raise")

    leg = _make_leg(1.30, 0.98, 0.94, 0.75)
    sig_atm = torch.tensor([0.15], dtype=torch.float64)
    rr25 = torch.tensor([0.02], dtype=torch.float64)
    bf_market = torch.tensor([0.008], dtype=torch.float64)

    def mock_flat_smile(K: Tensor, bf: Tensor) -> Tensor:
        return sig_atm + bf

    bf_smile = market_to_smile_strangle(
        bf_market=bf_market,
        rr=rr25,
        sigma_atm=sig_atm,
        delta=0.25,
        leg=leg,
        conv=DeltaConvention.FORWARD_PNA,
        smile_fn=mock_flat_smile,
    )

    assert torch.isfinite(bf_smile).all()
    assert abs(bf_smile.item() - bf_market.item()) < 1e-3


# ── Challenge 7: Calendar Monotonicity Checker Recall ──

def test_stress_calendar_monotonicity_checker() -> None:
    """Stress tests calendar monotonicity verification and violation detection."""
    T = torch.tensor([0.25, 0.50, 1.0, 2.0], dtype=torch.float64)
    M = 51

    # Case A: Monotonically increasing total variance
    w_valid = torch.stack([
        torch.full((M,), 0.04 * 0.25, dtype=torch.float64),
        torch.full((M,), 0.04 * 0.50, dtype=torch.float64),
        torch.full((M,), 0.04 * 1.00, dtype=torch.float64),
        torch.full((M,), 0.04 * 2.00, dtype=torch.float64),
    ])
    valid_flags = calendar_check(w_valid, T)
    assert bool(valid_flags.all().item())

    # Case B: Calendar arbitrage violation at T=1.0 (w decreases from T=0.50)
    w_invalid = w_valid.clone()
    w_invalid[2] = w_invalid[1] * 0.8  # total variance drops
    invalid_flags = calendar_check(w_invalid, T)
    assert not bool(invalid_flags[2].item())
