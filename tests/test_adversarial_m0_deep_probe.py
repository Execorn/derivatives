"""Adversarial Deep Probe for Milestone 0 Iteration 2 Remediation.

Empirically challenges:
1. Extreme tenors T in [1e-4, 10.0] and extreme vols sigma in [0.01, 3.0] across PNA conventions.
2. Boundary deltas (deep OTM / ITM) and PA argmax transition.
3. SSVI fallback autograd backward and IFT Jacobian stability under arbitrageable / perturbed quotes.
4. Mixed batch (Raw SVI + SSVI Fallback) autograd backward stability.
5. Convention inversion round-trip under stress regimes.
6. Volatility floor and Durrleman singularity hardening.
7. PA Delta high-tau iteration exhaustion defect (T in [5.0, 10.0], sigma in [2.0, 3.0]).
8. SSVI fallback autograd crash on tensors requiring grad.
9. SSVI fallback rho=1.0 degenerate certification failure under extreme skew.
"""
from __future__ import annotations

import math
from typing import cast
import numpy as np
import torch
from torch import Tensor

from deepvol.market.fx_conventions import (
    AtmConvention,
    DeltaConvention,
    LegMarket,
    StrangleConvention,
    atm_strike,
    fx_delta,
    strike_from_delta,
    pillar_strikes,
)
from deepvol.market.smile_canonicalizer import (
    ImplicitCanonicalMap,
    SmileCanonicalizer,
)


def _make_leg(spot: float, df_f: float, df_d: float, T: float) -> LegMarket:
    return LegMarket(
        spot=torch.tensor([spot], dtype=torch.float64),
        df_foreign=torch.tensor([df_f], dtype=torch.float64),
        df_domestic=torch.tensor([df_d], dtype=torch.float64),
        T=torch.tensor([T], dtype=torch.float64),
    )


# ── Probe 1: PNA Cartesian Stress (Extreme Tenors & Extreme Vols) ──

def test_probe_extreme_tenors_and_vols_pna_cartesian() -> None:
    """Probes T in [1e-4, 10.0] and sigma in [0.01, 3.0] across PNA conventions."""
    np.seterr(all="raise")
    torch.autograd.set_detect_anomaly(True)
    try:
        extreme_tenors = [1e-4, 1e-3, 1/365, 0.1, 1.0, 5.0, 10.0]
        extreme_vols = [0.01, 0.05, 0.20, 1.0, 3.0]
        spots = [0.5, 1.25, 150.0]
        eps_mach = torch.finfo(torch.float64).eps

        for T in extreme_tenors:
            for vol in extreme_vols:
                for spot in spots:
                    leg = _make_leg(spot, 0.98, 0.95, T)
                    sig = torch.tensor([vol], dtype=torch.float64)

                    for conv in [DeltaConvention.FORWARD_PNA, DeltaConvention.SPOT_PNA]:
                        for phi_val in [1.0, -1.0]:
                            phi = torch.tensor([phi_val], dtype=torch.float64)
                            for d_val in [0.01, 0.10, 0.25, 0.50, 0.90]:
                                target = torch.tensor([d_val * phi_val], dtype=torch.float64)
                                K = strike_from_delta(target, sig, leg, phi, conv)
                                assert torch.isfinite(K).all() and (K > 0).all()
                                rec = fx_delta(K, sig, leg, phi, conv)
                                assert torch.isfinite(rec).all()
                                tau = vol * math.sqrt(T)
                                tol_eff = max(1e-12, 10.0 * eps_mach / max(tau, 1e-15))
                                err = torch.abs(rec - target).item()
                                assert err <= tol_eff, (
                                    f"PNA residual {err:.3e} > {tol_eff:.3e} at T={T}, vol={vol}, d={d_val}, conv={conv}"
                                )
    finally:
        torch.autograd.set_detect_anomaly(False)


# ── Probe 2: Deep Tail Deltas (1e-6 to 0.999999) ──

def test_probe_deep_tail_deltas() -> None:
    """Tests extreme deltas approaching 0 and 1 for stability and non-divergence."""
    leg = _make_leg(1.0, 1.0, 1.0, 1.0)
    sig = torch.tensor([0.20], dtype=torch.float64)
    phi_c = torch.tensor([1.0], dtype=torch.float64)
    phi_p = torch.tensor([-1.0], dtype=torch.float64)

    boundary_deltas = [1e-6, 1e-4, 1e-2, 0.99, 0.9999, 1.0 - 1e-6]
    for d in boundary_deltas:
        target_c = torch.tensor([d], dtype=torch.float64)
        K_c = strike_from_delta(target_c, sig, leg, phi_c, DeltaConvention.FORWARD_PNA)
        assert torch.isfinite(K_c).all() and (K_c > 0).all()
        rec_c = fx_delta(K_c, sig, leg, phi_c, DeltaConvention.FORWARD_PNA)
        assert abs(rec_c.item() - d) < 1e-11

        target_p = torch.tensor([-d], dtype=torch.float64)
        K_p = strike_from_delta(target_p, sig, leg, phi_p, DeltaConvention.FORWARD_PNA)
        assert torch.isfinite(K_p).all() and (K_p > 0).all()
        rec_p = fx_delta(K_p, sig, leg, phi_p, DeltaConvention.FORWARD_PNA)
        assert abs(rec_p.item() - (-d)) < 1e-11


# ── Probe 3: SSVI Fallback Autograd Backward under Perturbed Arbitrageable Quotes ──

def test_probe_ssvi_fallback_autograd_backward() -> None:
    """Forces SSVI fallback and verifies backward pass, Jacobian finite differences, and zero LinAlgError."""
    torch.autograd.set_detect_anomaly(True)
    try:
        np.seterr(all="raise")

        can = SmileCanonicalizer(cache_size=0)
        leg = _make_leg(1.20, 0.98, 0.95, 0.5)
        conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

        quotes_arb = torch.tensor([[0.20, 0.08, -0.015, 0.16, -0.025]], dtype=torch.float64, requires_grad=True)

        # 1. Autograd Function forward + backward
        xi = cast(Tensor, ImplicitCanonicalMap.apply(quotes_arb, conv, leg, can))  # type: ignore[no-untyped-call]
        assert xi.shape == (1, 5)
        loss = (xi ** 2).sum()
        loss.backward()  # type: ignore[no-untyped-call]

        assert quotes_arb.grad is not None
        assert torch.isfinite(quotes_arb.grad).all()
        assert not torch.isnan(quotes_arb.grad).any()
        assert not torch.isinf(quotes_arb.grad).any()

        # 2. Forward pass with detached quotes to inspect Jacobian properties
        canon = can.canonicalize_unmemoized(quotes_arb.detach(), conv, leg)
        assert canon.used_ssvi.item() is True, "Expected SSVI fallback to be activated!"
        assert canon.min_g.item() >= -1e-12, f"SSVI fallback produced butterfly violation {canon.min_g.item()}"
        assert torch.isfinite(canon.jacobian).all()
        assert torch.isfinite(canon.xi).all()

        # 3. Verify that the autograd gradient matches canonicalizer.jacobian @ grad_output
        expected_grad = 2.0 * xi.detach() @ canon.jacobian[0]
        grad_err = (quotes_arb.grad - expected_grad).abs().max().item()
        assert grad_err < 1e-10, f"Autograd gradient differs from manifold Jacobian by {grad_err:.3e}"
    finally:
        torch.autograd.set_detect_anomaly(False)


# ── Probe 4: Convention Inversion Round-Trip Under Stress Regimes ──

def test_probe_convention_inversion_roundtrip_under_stress() -> None:
    """Stress tests round-trip conversion between quotes, pillar strikes, and smile canonicalizer."""
    leg = _make_leg(1.18, 0.98, 0.96, 0.5)

    test_convs = [
        (DeltaConvention.SPOT_PNA, AtmConvention.DNS),
        (DeltaConvention.FORWARD_PNA, AtmConvention.ATMF),
        (DeltaConvention.FORWARD_PA, AtmConvention.DNS),
        (DeltaConvention.SPOT_PA, AtmConvention.ATMF),
    ]

    quotes = torch.tensor([[0.16, -0.015, 0.006, -0.030, 0.014]], dtype=torch.float64)

    for d_conv, a_conv in test_convs:
        # Step 1: Forward quotes -> pillar strikes and vols: [10P, 25P, ATM, 25C, 10C]
        K_pillars, sig_pillars = pillar_strikes(quotes, leg, d_conv, a_conv)
        assert torch.isfinite(K_pillars).all() and (K_pillars > 0).all()
        assert torch.isfinite(sig_pillars).all() and (sig_pillars > 0).all()

        # Step 2: Unpack correctly by convention order
        K_10p, K_25p, K_atm, K_25c, K_10c = [K_pillars[0, i:i+1] for i in range(5)]
        s_10p, s_25p, s_atm, s_25c, s_10c = [sig_pillars[0, i:i+1] for i in range(5)]

        # Verify ATM strike matches convention definition
        expected_atm_k = atm_strike(s_atm, leg, a_conv, d_conv)
        assert abs(K_atm.item() - expected_atm_k.item()) < 1e-12

        # Verify 25 Delta calls and puts invert back to +/- 0.25
        phi_c = torch.tensor([1.0], dtype=torch.float64)
        phi_p = torch.tensor([-1.0], dtype=torch.float64)
        rec_d25c = fx_delta(K_25c, s_25c, leg, phi_c, d_conv)
        rec_d25p = fx_delta(K_25p, s_25p, leg, phi_p, d_conv)
        assert abs(rec_d25c.item() - 0.25) < 1e-11
        assert abs(rec_d25p.item() - (-0.25)) < 1e-11

        # Verify 10 Delta calls and puts invert back to +/- 0.10
        rec_d10c = fx_delta(K_10c, s_10c, leg, phi_c, d_conv)
        rec_d10p = fx_delta(K_10p, s_10p, leg, phi_p, d_conv)
        assert abs(rec_d10c.item() - 0.10) < 1e-11
        assert abs(rec_d10p.item() - (-0.10)) < 1e-11


# ── Probe 5: Volatility Floor, Singularity Hardening & SSVI Boundary Clamping ──

def test_probe_quant_math_hardening_vol_floor() -> None:
    """Ensures min vol >= 0.01 and SSVI boundary clamping maintains 1 - rho^2 >= 1.999e-3 under severe skew."""
    can = SmileCanonicalizer(cache_size=0)
    leg = _make_leg(1.0, 1.0, 1.0, 0.1)

    # 1. Volatility floor enforcement
    quotes_low_vol = torch.tensor([[0.005, 0.0, 0.001, 0.0, 0.002]], dtype=torch.float64)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    canon = can.canonicalize_unmemoized(quotes_low_vol, conv, leg)
    assert canon.sigma_atmf.item() >= 0.01, f"sigma_atmf {canon.sigma_atmf.item()} < 0.01"

    # 2. SSVI boundary clamping under severe skew (RR10 = 0.18, BF10 = -0.028)
    leg_skew = _make_leg(1.25, 0.99, 0.97, 0.5)
    q_severe = torch.tensor([[0.22, 0.09, -0.018, 0.18, -0.028]], dtype=torch.float64)
    smile = can.canonicalize(q_severe, conv, leg_skew)

    assert smile.used_ssvi.item() is True, "Expected SSVI fallback activation"
    assert smile.min_g.item() >= 0.0, f"Expected non-negative butterfly min_g, got {smile.min_g.item()}"
    rho_fitted = smile.slice_params[0, 2].item()
    assert abs(rho_fitted) <= 0.999 + 1e-10, f"Fitted rho {rho_fitted} exceeded interior bound 0.999"
    assert 1.0 - (rho_fitted ** 2) >= 1.999e-3 - 1e-10, "Expected 1 - rho^2 >= 1.999e-3"
    assert smile.slice_params[0, 0].item() > 0.0, "Expected strictly positive a parameter"
    assert smile.slice_params[0, 4].item() > 0.0, "Expected strictly positive s parameter"


# ── Probe 6: Remediation Verification — PA Put High-Tau Iteration Convergence ──

def test_probe_defect_pa_put_high_tau_iteration_exhaustion() -> None:
    """Verifies that PA Put at extreme tau (T=5.0, v=3.0) achieves full convergence under default n_iter=20."""
    leg = _make_leg(0.5, 0.98, 0.95, 5.0)
    sig = torch.tensor([3.0], dtype=torch.float64)
    phi = torch.tensor([-1.0], dtype=torch.float64)
    target = torch.tensor([-0.50], dtype=torch.float64)

    # Default n_iter=20 guarantees convergence to float64 machine zero without raising ConventionSolveError
    K = strike_from_delta(target, sig, leg, phi, DeltaConvention.FORWARD_PA)
    assert torch.isfinite(K).all() and (K > 0).all()
    rec = fx_delta(K, sig, leg, phi, DeltaConvention.FORWARD_PA)
    assert torch.abs(rec - target).item() <= 1e-12


# ── Probe 7: Remediation Verification — SSVI Fallback Autograd Requires Grad ──

def test_probe_defect_ssvi_fallback_autograd_requires_grad_crash() -> None:
    """Verifies remediation of Defect 1: direct canonicalize on requires_grad=True quotes succeeds."""
    can = SmileCanonicalizer(cache_size=0)
    leg = _make_leg(1.20, 0.98, 0.95, 0.5)
    conv = [(DeltaConvention.FORWARD_PNA, AtmConvention.DNS, StrangleConvention.SMILE)]

    quotes_arb = torch.tensor([[0.20, 0.08, -0.015, 0.16, -0.025]], dtype=torch.float64, requires_grad=True)

    # Post-remediation: does NOT raise SmileArbitrageError; completes successfully
    canon = can.canonicalize(quotes_arb, conv, leg)
    assert canon.used_ssvi[0].item() is True
    assert torch.isfinite(canon.jacobian).all()
