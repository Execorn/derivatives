from __future__ import annotations

import math
import pytest
import torch
from hypothesis import given, settings, strategies as st

from deepvol.market.fx_conventions import (
    AtmConvention,
    ConventionSolveError,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    atm_strike,
    fx_delta,
    market_to_smile_strangle,
    pa_call_delta_argmax,
    pillar_strikes,
    strike_from_delta,
)
from deepvol.market.fx_data import invert_gk_delta


def make_leg(spot: float = 1.25, df_f: float = 0.98, df_d: float = 0.95, T: float = 1.0) -> LegMarket:
    return LegMarket(
        spot=torch.tensor([spot], dtype=torch.float64),
        df_foreign=torch.tensor([df_f], dtype=torch.float64),
        df_domestic=torch.tensor([df_d], dtype=torch.float64),
        T=torch.tensor([T], dtype=torch.float64),
    )


# ── Test (a): Hypothesis Round Trip Δ -> K -> Δ ──

@settings(max_examples=50, deadline=None)
@given(
    sigma_val=st.floats(min_value=0.03, max_value=1.5),
    T_val=st.floats(min_value=1.0 / 365.0, max_value=5.0),
    abs_delta=st.floats(min_value=0.05, max_value=0.45),
    conv=st.sampled_from([
        DeltaConvention.FORWARD_PNA,
        DeltaConvention.SPOT_PNA,
        DeltaConvention.FORWARD_PA,
        DeltaConvention.SPOT_PA,
    ]),
    is_call=st.booleans(),
)
def test_hypothesis_delta_round_trip(
    sigma_val: float,
    T_val: float,
    abs_delta: float,
    conv: DeltaConvention,
    is_call: bool,
) -> None:
    leg = make_leg(spot=1.30, df_f=0.97, df_d=0.95, T=T_val)
    sigma = torch.tensor([sigma_val], dtype=torch.float64)
    phi_val = 1.0 if is_call else -1.0
    phi = torch.tensor([phi_val], dtype=torch.float64)
    target_delta_val = abs_delta if is_call else -abs_delta

    # Premium-adjusted call delta has a theoretical upper bound Delta_max(tau)
    if is_call and conv in (DeltaConvention.FORWARD_PA, DeltaConvention.SPOT_PA):
        K_max = pa_call_delta_argmax(sigma, leg)
        delta_max = fx_delta(K_max, sigma, leg, phi, conv).item()
        if target_delta_val > delta_max - 1e-4:
            # Target delta is near or above the theoretical ceiling: must raise ConventionSolveError
            target_delta = torch.tensor([target_delta_val], dtype=torch.float64)
            with pytest.raises(ConventionSolveError):
                strike_from_delta(target_delta, sigma, leg, phi, conv)
            return

    target_delta = torch.tensor([target_delta_val], dtype=torch.float64)
    K = strike_from_delta(target_delta, sigma, leg, phi, conv)
    delta_recovered = fx_delta(K, sigma, leg, phi, conv)

    diff = torch.abs(delta_recovered - target_delta).item()
    assert diff <= 1e-12, f"Round trip error {diff} > 1e-12 for conv={conv}, is_call={is_call}, sigma={sigma_val}, T={T_val}"


# ── Test (b): PA vs PNA 25Δ Call Strike Relative Shift at 1Y, σ=20% ──

def test_pa_vs_pna_25delta_call_shift() -> None:
    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([1.0], dtype=torch.float64),
        df_domestic=torch.tensor([1.0], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )
    sigma = torch.tensor([0.20], dtype=torch.float64)
    phi = torch.tensor([1.0], dtype=torch.float64)
    delta = torch.tensor([0.25], dtype=torch.float64)

    K_pna = strike_from_delta(delta, sigma, leg, phi, DeltaConvention.FORWARD_PNA).item()
    K_pa = strike_from_delta(delta, sigma, leg, phi, DeltaConvention.FORWARD_PA).item()

    rel_shift = (K_pa - K_pna) / K_pna
    expected_shift = -0.01912345
    # Relative shift must be -1.9% +/- 0.05%
    assert abs(rel_shift - (-0.019)) <= 0.0005, f"Shift {rel_shift:.6f} outside [-1.95%, -1.85%]"
    assert abs(rel_shift - expected_shift) <= 1e-5


# ── Test (c): Agreement with invert_gk_delta on Overlapping Conventions ──

@pytest.mark.parametrize("conv,gk_conv", [
    (DeltaConvention.FORWARD_PNA, "forward_pna"),
    (DeltaConvention.SPOT_PNA, "spot_pna"),
    (DeltaConvention.FORWARD_PA, "forward_pa"),
    (DeltaConvention.SPOT_PA, "spot_pa"),
])
@pytest.mark.parametrize("is_call", [True, False])
def test_agreement_with_invert_gk_delta(conv: DeltaConvention, gk_conv: str, is_call: bool) -> None:
    r_d = 0.03
    r_f = 0.01
    T = 1.0
    spot = 100.0
    df_f = math.exp(-r_f * T)
    df_d = math.exp(-r_d * T)
    F = spot * df_f / df_d
    vol = 0.20

    leg = LegMarket(
        spot=torch.tensor([spot], dtype=torch.float64),
        df_foreign=torch.tensor([df_f], dtype=torch.float64),
        df_domestic=torch.tensor([df_d], dtype=torch.float64),
        T=torch.tensor([T], dtype=torch.float64),
    )
    sigma = torch.tensor([vol], dtype=torch.float64)
    phi = torch.tensor([1.0 if is_call else -1.0], dtype=torch.float64)
    opt_type = "call" if is_call else "put"
    delta_val = 0.25 if is_call else -0.25

    K_gk = invert_gk_delta(
        F=F,
        delta=delta_val,
        T=T,
        r_d=r_d,
        r_f=r_f,
        vol=vol,
        option_type=opt_type,
        delta_type=gk_conv,
    )

    K_tensor = strike_from_delta(
        delta=torch.tensor([delta_val], dtype=torch.float64),
        sigma=sigma,
        leg=leg,
        phi=phi,
        conv=conv,
    ).item()

    rel_err = abs(K_tensor - K_gk) / K_gk
    assert rel_err <= 1e-10, f"Relative error {rel_err} > 1e-10 for conv={conv}, is_call={is_call}"


# ── Test (d): DNS Straddle Delta = 0 to 10^-12 ──

@pytest.mark.parametrize("conv", [
    DeltaConvention.FORWARD_PNA,
    DeltaConvention.SPOT_PNA,
    DeltaConvention.FORWARD_PA,
    DeltaConvention.SPOT_PA,
])
def test_dns_straddle_delta_zero(conv: DeltaConvention) -> None:
    leg = make_leg(spot=1.20, df_f=0.98, df_d=0.96, T=1.5)
    sigma = torch.tensor([0.22], dtype=torch.float64)

    K_dns = atm_strike(sigma, leg, AtmConvention.DNS, conv)
    delta_c = fx_delta(K_dns, sigma, leg, torch.tensor([1.0], dtype=torch.float64), conv)
    delta_p = fx_delta(K_dns, sigma, leg, torch.tensor([-1.0], dtype=torch.float64), conv)

    net_delta = (delta_c + delta_p).item()
    assert abs(net_delta) <= 1e-12, f"DNS straddle delta {net_delta} > 1e-12 for conv={conv}"


# ── Test (e): MARKET Strangle Inverted then Re-priced Reproduces Broker Premium ──

def test_market_strangle_inversion() -> None:
    leg = make_leg(spot=100.0, df_f=0.98, df_d=0.95, T=1.0)
    sigma_atm = torch.tensor([0.15], dtype=torch.float64)
    bf_market = torch.tensor([0.012], dtype=torch.float64)
    rr = torch.tensor([0.008], dtype=torch.float64)
    delta = 0.25
    conv = DeltaConvention.FORWARD_PNA

    # A smooth smile function representing market smile given candidate bf
    def smile_fn(K: torch.Tensor, bf: torch.Tensor) -> torch.Tensor:
        F = leg.forward
        x = torch.log(K / F)
        # quadratic smile in log-moneyness with curvature proportional to bf
        return sigma_atm + bf + 0.5 * rr * torch.tanh(x) + 0.8 * bf * (x ** 2)

    bf_smile = market_to_smile_strangle(
        bf_market=bf_market,
        rr=rr,
        sigma_atm=sigma_atm,
        delta=delta,
        leg=leg,
        conv=conv,
        smile_fn=smile_fn,
    )

    # Re-pricing verification:
    # 1. Broker strangle premium priced at single vol sigma_mkt = sigma_atm + bf_market
    sigma_mkt = sigma_atm + bf_market
    K_c = strike_from_delta(torch.tensor([delta], dtype=torch.float64), sigma_mkt, leg, torch.tensor([1.0], dtype=torch.float64), conv)
    K_p = strike_from_delta(torch.tensor([-delta], dtype=torch.float64), sigma_mkt, leg, torch.tensor([-1.0], dtype=torch.float64), conv)

    F = leg.forward
    T = leg.T
    d1_c = (torch.log(F / K_c) + 0.5 * (sigma_mkt ** 2) * T) / (sigma_mkt * torch.sqrt(T))
    d2_c = d1_c - sigma_mkt * torch.sqrt(T)
    call_mkt = leg.df_domestic * (F * torch.special.ndtr(d1_c) - K_c * torch.special.ndtr(d2_c))

    d1_p = (torch.log(F / K_p) + 0.5 * (sigma_mkt ** 2) * T) / (sigma_mkt * torch.sqrt(T))
    d2_p = d1_p - sigma_mkt * torch.sqrt(T)
    put_mkt = leg.df_domestic * (K_p * torch.special.ndtr(-d2_p) - F * torch.special.ndtr(-d1_p))
    prem_broker = (call_mkt + put_mkt).item()

    # 2. Smile strangle premium evaluated at strikes K_c, K_p on fitted smile
    vol_c = smile_fn(K_c, bf_smile)
    vol_p = smile_fn(K_p, bf_smile)

    d1_c_s = (torch.log(F / K_c) + 0.5 * (vol_c ** 2) * T) / (vol_c * torch.sqrt(T))
    d2_c_s = d1_c_s - vol_c * torch.sqrt(T)
    call_smile = leg.df_domestic * (F * torch.special.ndtr(d1_c_s) - K_c * torch.special.ndtr(d2_c_s))

    d1_p_s = (torch.log(F / K_p) + 0.5 * (vol_p ** 2) * T) / (vol_p * torch.sqrt(T))
    d2_p_s = d1_p_s - vol_p * torch.sqrt(T)
    put_smile = leg.df_domestic * (K_p * torch.special.ndtr(-d2_p_s) - F * torch.special.ndtr(-d1_p_s))
    prem_smile = (call_smile + put_smile).item()

    assert abs(prem_smile - prem_broker) <= 1e-10, f"Market strangle re-priced error {abs(prem_smile - prem_broker)} > 1e-10"


# ── Test (f): strike_from_delta Traced Graph Succeeds Under torch.compile(fullgraph=True) ──

def test_strike_from_delta_compile_fullgraph() -> None:
    leg = LegMarket(
        spot=torch.tensor([1.25, 1.30], dtype=torch.float64),
        df_foreign=torch.tensor([0.98, 0.97], dtype=torch.float64),
        df_domestic=torch.tensor([0.95, 0.94], dtype=torch.float64),
        T=torch.tensor([1.0, 1.5], dtype=torch.float64),
    )
    sigma = torch.tensor([0.18, 0.22], dtype=torch.float64)
    delta = torch.tensor([0.25, 0.25], dtype=torch.float64)
    phi = torch.tensor([1.0, 1.0], dtype=torch.float64)

    def solve_fn(d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return strike_from_delta(d, s, leg, phi, DeltaConvention.FORWARD_PA)

    compiled_solve = torch.compile(solve_fn, fullgraph=True)
    out = compiled_solve(delta, sigma)
    assert out.shape == (2,)
    assert torch.all(out > 0)


# ── Test Pillar Strikes Shape and Consistency ──

def test_pillar_strikes() -> None:
    leg = LegMarket(
        spot=torch.tensor([1.20, 100.0], dtype=torch.float64),
        df_foreign=torch.tensor([0.98, 0.99], dtype=torch.float64),
        df_domestic=torch.tensor([0.95, 0.97], dtype=torch.float64),
        T=torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    quotes = torch.tensor([
        [0.15, -0.010, 0.005, -0.018, 0.012],
        [0.20, 0.015, 0.008, 0.025, 0.016],
    ], dtype=torch.float64)

    K, sigmas = pillar_strikes(quotes, leg, DeltaConvention.FORWARD_PNA, AtmConvention.DNS)
    assert K.shape == (2, 5)
    assert sigmas.shape == (2, 5)
    # Check monotonic strike ordering for standard currency pairs K_10P < K_25P < K_ATM < K_25C < K_10C
    for s_idx in range(2):
        assert K[s_idx, 0] < K[s_idx, 1] < K[s_idx, 2] < K[s_idx, 3] < K[s_idx, 4]
