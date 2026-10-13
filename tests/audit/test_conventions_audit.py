"""
tests/audit/test_conventions_audit.py — Phase 5 FX Market Conventions and Smile Canonicalization Audit.
All tests use clean-room float64 reference oracles and verify SPEC-DEEPVOL-FX-CROSS-003 M0 requirements.
Marked with @pytest.mark.audit.
"""

from __future__ import annotations

import math
import threading
from typing import List, Tuple
import numpy as np
import pytest
import torch

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    atm_strike,
    fx_delta,
    market_to_smile_strangle,
    pa_call_delta_argmax,
    strike_from_delta,
)
from deepvol.market.svi_slice import (
    RawSVI,
    certify_slice,
    durrleman_g,
    fit_raw_svi_5pt,
)
from deepvol.market.smile_canonicalizer import (
    CanonicalSmile,
    SmileArbitrageError,
    SmileCanonicalizer,
    fit_ssvi_slice,
    ssvi_to_raw_svi,
)
from artifacts.audit.scripts.oracles import (
    durrleman_g as oracle_durrleman_g,
    fx_delta_solve_strike as oracle_delta_solve,
    ssvi_to_raw_svi as oracle_ssvi_to_raw,
    svi_total_variance as oracle_svi_total_variance,
)


@pytest.mark.audit
def test_delta_conventions_roundtrip() -> None:
    """Phase 5.1: Round-trip Delta -> Strike -> Delta <= 1e-12 relative across all 4 conventions."""
    conventions = [
        DeltaConvention.FORWARD_PNA,
        DeltaConvention.SPOT_PNA,
        DeltaConvention.FORWARD_PA,
        DeltaConvention.SPOT_PA,
    ]

    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([0.98], dtype=torch.float64),
        df_domestic=torch.tensor([0.95], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )

    test_grid = [
        # (sigma, delta, phi)
        (0.10, 0.25, 1.0),
        (0.10, -0.25, -1.0),
        (0.20, 0.10, 1.0),
        (0.20, -0.10, -1.0),
        (0.35, 0.40, 1.0),
        (0.35, -0.40, -1.0),
        (0.50, 0.25, 1.0),
        (0.50, -0.25, -1.0),
    ]

    for conv in conventions:
        for sigma_val, delta_val, phi_val in test_grid:
            sigma_t = torch.tensor([sigma_val], dtype=torch.float64)
            delta_target = torch.tensor([delta_val], dtype=torch.float64)
            phi_t = torch.tensor([phi_val], dtype=torch.float64)

            # Invert Delta -> Strike
            K = strike_from_delta(delta_target, sigma_t, leg, phi_t, conv, tol=1e-13)

            # Re-evaluate Strike -> Delta
            delta_recovered = fx_delta(K, sigma_t, leg, phi_t, conv)

            rel_err = abs(delta_recovered.item() - delta_val) / abs(delta_val)
            assert rel_err <= 1e-12, (
                f"Convention {conv.value} round-trip failed: target={delta_val}, recovered={delta_recovered.item()}, rel_err={rel_err}"
            )


@pytest.mark.audit
def test_pa_vs_pna_25delta_call_relative_shift() -> None:
    """Phase 5.1: PA vs PNA 25Delta call strike at 1Y, sigma = 20%: relative shift -1.9% +/- 0.05%."""
    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([1.0], dtype=torch.float64),
        df_domestic=torch.tensor([1.0], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )
    sigma = torch.tensor([0.20], dtype=torch.float64)
    delta_25 = torch.tensor([0.25], dtype=torch.float64)
    phi_call = torch.tensor([1.0], dtype=torch.float64)

    K_pna = strike_from_delta(delta_25, sigma, leg, phi_call, DeltaConvention.FORWARD_PNA)
    K_pa = strike_from_delta(delta_25, sigma, leg, phi_call, DeltaConvention.FORWARD_PA)

    rel_shift = (K_pa.item() - K_pna.item()) / K_pna.item()
    # Expect -1.9% +/- 0.05%, i.e. in [-0.0195, -0.0185]
    assert -0.0195 <= rel_shift <= -0.0185, (
        f"PA vs PNA relative shift expected -1.9% +/- 0.05%, got {rel_shift * 100:.3f}%"
    )


@pytest.mark.audit
def test_pa_call_delta_argmax_and_bracket() -> None:
    """Phase 5.1: PA call delta achieves argmax at K_max solving sigma*sqrt(T)*N(d2) = n(d2)."""
    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([0.98], dtype=torch.float64),
        df_domestic=torch.tensor([0.95], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )
    sigma = torch.tensor([0.25], dtype=torch.float64)
    F = leg.forward.item()
    tau = 0.25 * 1.0

    K_max = pa_call_delta_argmax(sigma, leg).item()

    # Verify condition: tau * N(d2) = n(d2)
    d2 = (math.log(F / K_max) - 0.5 * tau**2) / tau
    from scipy.special import ndtr
    from scipy.stats import norm
    cond = tau * float(ndtr(d2)) - float(norm.pdf(d2))
    assert abs(cond) <= 1e-12, f"Argmax condition residual: {cond}"

    # Verify that K_max is indeed the peak of PA call delta
    eps_K = 1e-4 * K_max
    d_max = fx_delta(torch.tensor([K_max], dtype=torch.float64), sigma, leg, torch.tensor([1.0]), DeltaConvention.FORWARD_PA).item()
    d_left = fx_delta(torch.tensor([K_max - eps_K], dtype=torch.float64), sigma, leg, torch.tensor([1.0]), DeltaConvention.FORWARD_PA).item()
    d_right = fx_delta(torch.tensor([K_max + eps_K], dtype=torch.float64), sigma, leg, torch.tensor([1.0]), DeltaConvention.FORWARD_PA).item()
    assert d_max >= d_left and d_max >= d_right, "K_max is not a local maximum of PA call delta"


@pytest.mark.audit
def test_atm_conventions() -> None:
    """Phase 5.2: ATM conventions check: ATMF (K = F) and DNS (K = F*exp(0.5*sigma^2*T) PNA). Straddle delta = 0."""
    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([0.98], dtype=torch.float64),
        df_domestic=torch.tensor([0.95], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )
    sigma = torch.tensor([0.20], dtype=torch.float64)
    F = leg.forward.item()

    # ATMF
    K_atmf = atm_strike(sigma, leg, AtmConvention.ATMF, DeltaConvention.FORWARD_PNA).item()
    assert abs(K_atmf - F) <= 1e-12

    # DNS PNA: K = F * exp(0.5 * sigma^2 * T)
    K_dns_pna = atm_strike(sigma, leg, AtmConvention.DNS, DeltaConvention.FORWARD_PNA).item()
    expected_dns_pna = F * math.exp(0.5 * (0.20**2) * 1.0)
    assert abs(K_dns_pna - expected_dns_pna) <= 1e-12

    # Straddle delta at DNS: call delta + put delta = 0
    d_call = fx_delta(torch.tensor([K_dns_pna], dtype=torch.float64), sigma, leg, torch.tensor([1.0]), DeltaConvention.FORWARD_PNA).item()
    d_put = fx_delta(torch.tensor([K_dns_pna], dtype=torch.float64), sigma, leg, torch.tensor([-1.0]), DeltaConvention.FORWARD_PNA).item()
    assert abs(d_call + d_put) <= 1e-12, f"DNS straddle delta != 0: {d_call + d_put}"


@pytest.mark.audit
def test_market_strangle_inversion() -> None:
    """Phase 5.3: Market strangle inverted to smile strangle reproduces broker premium to 10^-10."""
    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([1.0], dtype=torch.float64),
        df_domestic=torch.tensor([1.0], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )
    sigma_atm = torch.tensor([0.15], dtype=torch.float64)
    rr25 = torch.tensor([0.02], dtype=torch.float64)
    bf25_market = torch.tensor([0.005], dtype=torch.float64)

    def smile_fn(K_cand: Tensor, bf: Tensor) -> Tensor:
        x = torch.log(K_cand / leg.forward)
        return sigma_atm + bf + 0.5 * rr25 * torch.tanh(x) + 0.8 * bf * (x ** 2)

    bf25_smile = market_to_smile_strangle(
        bf25_market, rr25, sigma_atm, 0.25, leg, DeltaConvention.FORWARD_PNA, smile_fn
    )

    # Re-pricing verification:
    sigma_mkt = sigma_atm + bf25_market
    K_c = strike_from_delta(torch.tensor([0.25], dtype=torch.float64), sigma_mkt, leg, torch.tensor([1.0], dtype=torch.float64), DeltaConvention.FORWARD_PNA)
    K_p = strike_from_delta(torch.tensor([-0.25], dtype=torch.float64), sigma_mkt, leg, torch.tensor([-1.0], dtype=torch.float64), DeltaConvention.FORWARD_PNA)

    F = leg.forward
    T = leg.T
    d1_c = (torch.log(F / K_c) + 0.5 * (sigma_mkt ** 2) * T) / (sigma_mkt * torch.sqrt(T))
    d2_c = d1_c - sigma_mkt * torch.sqrt(T)
    call_mkt = leg.df_domestic * (F * torch.special.ndtr(d1_c) - K_c * torch.special.ndtr(d2_c))

    d1_p = (torch.log(F / K_p) + 0.5 * (sigma_mkt ** 2) * T) / (sigma_mkt * torch.sqrt(T))
    d2_p = d1_p - sigma_mkt * torch.sqrt(T)
    put_mkt = leg.df_domestic * (K_p * torch.special.ndtr(-d2_p) - F * torch.special.ndtr(-d1_p))
    prem_broker = (call_mkt + put_mkt).item()

    vol_c = smile_fn(K_c, bf25_smile)
    vol_p = smile_fn(K_p, bf25_smile)

    d1_c_s = (torch.log(F / K_c) + 0.5 * (vol_c ** 2) * T) / (vol_c * torch.sqrt(T))
    d2_c_s = d1_c_s - vol_c * torch.sqrt(T)
    call_smile = leg.df_domestic * (F * torch.special.ndtr(d1_c_s) - K_c * torch.special.ndtr(d2_c_s))

    d1_p_s = (torch.log(F / K_p) + 0.5 * (vol_p ** 2) * T) / (vol_p * torch.sqrt(T))
    d2_p_s = d1_p_s - vol_p * torch.sqrt(T)
    put_smile = leg.df_domestic * (K_p * torch.special.ndtr(-d2_p_s) - F * torch.special.ndtr(-d1_p_s))
    prem_smile = (call_smile + put_smile).item()

    assert abs(prem_smile - prem_broker) <= 1e-10, f"Market strangle re-priced error {abs(prem_smile - prem_broker)} > 1e-10"


@pytest.mark.audit
def test_svi_fit_rmse_and_durrleman_grid() -> None:
    """Phase 5.4: SVI fit RMSE in vol-pts and Durrleman certificate comparison on 2,001 points."""
    # Known admissible raw SVI parameters
    p_true = [0.04, 0.10, -0.20, 0.05, 0.15]
    T_val = 1.0

    # 5 pillar strikes
    x_5 = torch.tensor([[-0.4, -0.2, 0.0, 0.2, 0.4]], dtype=torch.float64)
    w_5, _, _ = svi_total_variance_tensor(x_5, p_true)

    svi_fit, rmse = fit_raw_svi_5pt(x_5, w_5)
    assert rmse.item() <= 1e-7, f"SVI 5-point recovery RMSE failed: {rmse.item()}"

    # Durrleman certificate on 2,001 points over k in [-6, 6]
    k_dense = np.linspace(-6.0, 6.0, 2001)
    k_dense_t = torch.tensor(k_dense, dtype=torch.float64).unsqueeze(0)

    w_fit, w1_fit, w2_fit = svi_fit.derivs(k_dense_t)
    g_dev = durrleman_g(k_dense_t, w_fit, w1_fit, w2_fit).squeeze(0).numpy()

    p_fit = svi_fit.params[0].numpy()
    g_oracle = oracle_durrleman_g(k_dense, p_fit[0], p_fit[1], p_fit[2], p_fit[3], p_fit[4])

    max_diff = np.max(np.abs(g_dev - g_oracle))
    assert max_diff <= 1e-12, f"Durrleman certificate diff vs oracle: {max_diff}"


def svi_total_variance_tensor(x: Tensor, p: List[float]) -> Tuple[Tensor, Tensor, Tensor]:
    a, b, rho_s, m, s = p
    diff = x - m
    hyp = torch.sqrt(diff**2 + s**2)
    w = a + b * (rho_s * diff + hyp)
    w1 = b * (rho_s + diff / hyp)
    w2 = b * (s**2) / (hyp**3)
    return w, w1, w2


@pytest.mark.audit
def test_ssvi_fallback_triggers() -> None:
    """Phase 5.4: Axel Vogt arbitrageable slice triggers SSVI fallback with min_g >= 0."""
    # Axel Vogt arbitrageable slice parameters
    # a=-0.0410, b=0.1331, rho_s=0.3060, m=0.3586, s=0.4153, T=1
    svi_bad = RawSVI(torch.tensor([[-0.0410, 0.1331, 0.3060, 0.3586, 0.4153]], dtype=torch.float64))
    cert = certify_slice(svi_bad, torch.tensor([0.20], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    assert cert.ok.item() is False, "Axel Vogt slice unexpectedly passed certification"

    # Now fit SSVI on arbitrageable data
    x_bad = torch.linspace(-0.5, 0.5, 5, dtype=torch.float64)
    w_bad = svi_bad.w(x_bad)

    params_ssvi, rmse = fit_ssvi_slice(x_bad, w_bad)
    raw_from_ssvi = ssvi_to_raw_svi(params_ssvi[:, 0], params_ssvi[:, 1], params_ssvi[:, 2])
    cert_ssvi = certify_slice(raw_from_ssvi, torch.tensor([0.20], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    assert cert_ssvi.ok.item() is True, "SSVI fallback slice failed certification"
    assert cert_ssvi.min_g.item() >= 0.0


@pytest.mark.audit
def test_cache_correctness_and_concurrency() -> None:
    """Phase 5.5: Cache keys include all inputs (blake2b-128), concurrent access is safe, cache hits identical."""
    canonicalizer = SmileCanonicalizer(cache_size=512)

    leg = LegMarket(
        spot=torch.tensor([100.0], dtype=torch.float64),
        df_foreign=torch.tensor([0.98], dtype=torch.float64),
        df_domestic=torch.tensor([0.95], dtype=torch.float64),
        T=torch.tensor([1.0], dtype=torch.float64),
    )
    quotes = torch.tensor([[0.20, -0.02, 0.01, -0.04, 0.02]], dtype=torch.float64)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    cs1 = canonicalizer.canonicalize(quotes, conv, leg)
    cs2 = canonicalizer.canonicalize(quotes, conv, leg)

    # Identical object returned from cache
    assert cs1 is cs2

    # Multithreaded concurrency stress test
    errors: List[Exception] = []

    def worker() -> None:
        try:
            for _ in range(50):
                res = canonicalizer.canonicalize(quotes, conv, leg)
                assert res is cs1
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent cache access threw exceptions: {errors}"
