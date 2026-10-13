"""tests/test_adversarial_cross_m3_stress.py — Adversarial Stress & Boundary Verification for Milestone 3.

Empirical Challenger verification suite probing:
1. Gate G3.1: 720-scenario Cartesian stress matrix under np.seterr(all="raise") & detect_anomaly(True).
2. Gate G3.4: Taper continuity limits across |z| in {5.0, 7.0} for V, Delta, Gamma (<= 1e-10 relative error).
3. Deep-wing intrinsic & Put-Call parity on XAU/RUB K=10 RUB (z approx +45) and K=1e7 RUB (z approx -45).
4. Butterfly arbitrage injection detection via cross_slice_audit.
5. Calendar arbitrage injection detection reproduction (auditing multi-expiry slices).
6. Model risk guardian audit routing verification (testing OUTPUT_AUDIT_FAILED failover).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import AnalyticTriangulated, Col, greeks_from_outputs
from deepvol.models.fx_cross_safeguards import (
    cross_slice_audit,
    smooth_time_floor,
    smooth_vol_floor,
    taper_jet,
    wing_slope_certificate,
    z_squash,
)
from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine, ResidualCrossVolNet
from deepvol.mrm.fx_cross_guardian import (
    FallbackReasonEnum,
    FXCrossGuardian,
    PricingEngineEnum,
)


def _make_clean_test_state() -> Tensor:
    """Creates a standard valid XAU/RUB 34-column market state tensor (1, 34)."""
    X = torch.zeros(1, 34, dtype=torch.float64)
    F1 = 2650.0
    F2 = 95.0
    F_val = F1 * F2
    X[0, Col.F] = F_val
    X[0, Col.K] = F_val  # ATM
    X[0, Col.T] = 1.0
    X[0, Col.T_SET] = 1.0
    X[0, Col.D] = 0.85
    X[0, Col.G] = 0.98 / 0.85  # D_A / D_C
    X[0, Col.RHO] = -0.20
    X[0, Col.PHI] = 1.0  # Call

    # Leg 1 SVI params [a, b, rho_s, m, s]
    X[0, Col.SVI1 : Col.SVI1 + 5] = torch.tensor([0.04, 0.10, -0.20, 0.0, 0.10], dtype=torch.float64)
    # Leg 2 SVI params
    X[0, Col.SVI2 : Col.SVI2 + 5] = torch.tensor([0.09, 0.15, 0.25, 0.0, 0.12], dtype=torch.float64)

    # Canonical features xi1, xi2
    X[0, Col.XI1 : Col.XI1 + 5] = torch.tensor([0.20, -0.05, 0.02, -0.08, 0.03], dtype=torch.float64)
    X[0, Col.XI2 : Col.XI2 + 5] = torch.tensor([0.30, 0.08, 0.03, 0.12, 0.05], dtype=torch.float64)

    X[0, Col.DELTA1] = 0.0
    X[0, Col.DELTA2] = 0.0
    X[0, Col.R_C] = 0.15
    X[0, Col.R_A] = 0.02
    X[0, Col.R_SET] = 0.15
    X[0, Col.NN_GATE] = 1.0
    return X


def test_adversarial_g31_cartesian_stress_matrix() -> None:
    """Gate G3.1: Full 720-scenario Cartesian stress matrix.

    Asserts zero NaN, zero Inf, zero negative volatilities, and zero exceptions
    under torch.autograd.set_detect_anomaly(True) and host np.seterr(all="raise").
    """
    old_err = np.seterr(all="raise")
    try:
        net = ResidualCrossVolNet(width=64, depth=3, members=1)
        engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)

        sk_ratios = [1e-6, 1e-5, 1.0, 1e5, 1e6]
        T_values = [1e-5, 1e-4, 1 / 365, 5.0]
        vols = [1e-4, 0.01, 3.0]
        rhos = [-0.999, 0.0, 0.999]
        parities = [1.0, -1.0]

        base_state = _make_clean_test_state().squeeze(0)
        F_base = float(base_state[Col.F].item())

        rows = []
        for sk in sk_ratios:
            for t_val in T_values:
                for vol_val in vols:
                    for rho_val in rhos:
                        for phi_val in parities:
                            row = base_state.clone()
                            row[Col.K] = F_base / sk
                            row[Col.T] = t_val
                            row[Col.T_SET] = max(t_val, 1e-4)
                            row[Col.RHO] = rho_val
                            row[Col.PHI] = phi_val
                            row[Col.SVI1] = (vol_val ** 2) * max(t_val, 1e-4)
                            row[Col.SVI1 + 1] = 0.01
                            row[Col.SVI1 + 4] = 0.10
                            row[Col.SVI2] = (vol_val ** 2) * max(t_val, 1e-4)
                            row[Col.SVI2 + 1] = 0.01
                            row[Col.SVI2 + 4] = 0.10
                            rows.append(row)

        assert len(rows) == 360, f"Expected 360 parameter combinations, got {len(rows)}"

        X_batch = torch.stack(rows, dim=0).requires_grad_(True)

        with torch.autograd.set_detect_anomaly(True):
            out = engine(X_batch)

            # Outputs finite checks
            assert torch.isfinite(out.V).all(), "Non-finite price output in Cartesian matrix"
            assert torch.isfinite(out.sigma).all(), "Non-finite sigma in Cartesian matrix"
            assert (out.sigma >= 0.01).all(), "Volatility fell below 0.01 soft floor"
            assert (out.sigma > 0).all(), "Negative volatility in Cartesian matrix"
            assert torch.isfinite(out.s_k).all(), "Non-finite s_k in Cartesian matrix"
            assert torch.isfinite(out.s_kk).all(), "Non-finite s_kk in Cartesian matrix"

            # Greeks evaluation under both dynamics (360 * 2 = 720 scenarios)
            for dyn in ("STICKY_MONEYNESS", "STICKY_STRIKE"):
                greeks = greeks_from_outputs(X_batch, out, dynamics=dyn)
                for g_name, g_val in greeks.items():
                    assert torch.isfinite(g_val).all(), f"Non-finite Greek {g_name} under {dyn}"

            # Full autograd backward gradient check
            loss = out.V.sum()
            loss.backward()
            assert X_batch.grad is not None
            assert torch.isfinite(X_batch.grad).all(), "Non-finite backward gradient in Cartesian matrix"

    finally:
        np.seterr(**old_err)


def test_adversarial_g34_taper_continuity_limits() -> None:
    """Gate G3.4: Taper continuity one-sided limits across |z| in {5.0, 7.0}.

    Verifies relative jump <= 1e-10 for V, Delta, Gamma under both dynamics.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)
    X = _make_clean_test_state()
    F_val = float(X[0, Col.F].item())

    # Reference cross vol
    p1 = X[0, Col.SVI1 : Col.SVI1 + 5]
    p2 = X[0, Col.SVI2 : Col.SVI2 + 5]
    a1, b1, rho_s1, m1, s1 = [float(x.item()) for x in p1]
    a2, b2, rho_s2, m2, s2 = [float(x.item()) for x in p2]
    w1_0 = a1 + b1 * (rho_s1 * (-m1) + (m1 * m1 + s1 * s1) ** 0.5)
    w2_0 = a2 + b2 * (rho_s2 * (-m2) + (m2 * m2 + s2 * s2) ** 0.5)
    sig1_0 = (w1_0 / 1.0) ** 0.5
    sig2_0 = (w2_0 / 1.0) ** 0.5
    sig_ref = (sig1_0 ** 2 + sig2_0 ** 2 + 2 * (-0.2) * sig1_0 * sig2_0) ** 0.5

    eps = 1e-12
    boundaries = [5.0, -5.0, 7.0, -7.0]

    for b in boundaries:
        sgn = 1.0 if b > 0 else -1.0
        z_in = b - sgn * eps
        z_out = b + sgn * eps

        for dyn in ("STICKY_MONEYNESS", "STICKY_STRIKE"):
            X[0, Col.K] = F_val * torch.exp(torch.tensor(z_in * sig_ref, dtype=torch.float64))
            out_in = engine(X)
            gr_in = greeks_from_outputs(X, out_in, dynamics=dyn)

            X[0, Col.K] = F_val * torch.exp(torch.tensor(z_out * sig_ref, dtype=torch.float64))
            out_out = engine(X)
            gr_out = greeks_from_outputs(X, out_out, dynamics=dyn)

            rel_V = abs(out_in.V.item() - out_out.V.item()) / out_in.V.item()
            rel_delta = abs(gr_in["delta"].item() - gr_out["delta"].item()) / max(abs(gr_in["delta"].item()), 1e-12)
            rel_gamma = abs(gr_in["gamma"].item() - gr_out["gamma"].item()) / max(abs(gr_in["gamma"].item()), 1e-12)

            assert rel_V <= 1e-10, f"Gate G3.4 V continuity failure at {b} ({dyn}): rel_V = {rel_V:.2e} > 1e-10"
            assert rel_delta <= 1e-10, f"Gate G3.4 Delta continuity failure at {b} ({dyn}): rel_delta = {rel_delta:.2e} > 1e-10"
            assert rel_gamma <= 1e-10, f"Gate G3.4 Gamma continuity failure at {b} ({dyn}): rel_gamma = {rel_gamma:.2e} > 1e-10"


def test_adversarial_deep_wing_intrinsic_parity() -> None:
    """Deep-Wing Intrinsic & Parity Probe.

    Verifies:
    1. K = 10 RUB Call on XAU/RUB (z approx +45):
       V = D(F - K) to 1e-10 relative error, Delta = Dg to 1e-12 absolute error, chi == 0.0.
    2. K = 10 RUB Put on XAU/RUB (deep OTM): V = 0.0, chi == 0.0.
    3. Put-Call parity V_call - V_put == D(F - K) to 1e-12.
    4. Bitwise equality to AnalyticTriangulated at |z| >= 7.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)
    analytic = AnalyticTriangulated()

    X_call = _make_clean_test_state()
    F_val = float(X_call[0, Col.F].item())  # 251750.0
    K_val = 10.0
    D_val = float(X_call[0, Col.D].item())  # 0.85
    g_val = float(X_call[0, Col.G].item())  # 0.98 / 0.85

    X_call[0, Col.K] = K_val
    X_call[0, Col.PHI] = 1.0  # Call

    X_put = X_call.clone()
    X_put[0, Col.PHI] = -1.0  # Put

    out_call = engine(X_call)
    out_put = engine(X_put)
    out_ana_call = analytic(X_call)

    # 1. Price matches D(F-K) to 1e-10 relative
    expected_call = D_val * (F_val - K_val)
    price_call = out_call.V.item()
    rel_err_call = abs(price_call - expected_call) / expected_call
    assert rel_err_call <= 1e-10, f"Deep Call relative error {rel_err_call} > 1e-10"

    # 2. Delta matches Dg to 1e-12
    gr_call = greeks_from_outputs(X_call, out_call, dynamics="STICKY_MONEYNESS")
    delta_call = gr_call["delta"].item()
    expected_delta = D_val * g_val
    assert abs(delta_call - expected_delta) <= 1e-12, f"Delta error {abs(delta_call - expected_delta)} > 1e-12"

    # 3. NN contribution exactly 0.0
    assert float(out_call.chi.item()) == 0.0
    assert float(out_put.chi.item()) == 0.0

    # 4. Put price is 0.0
    assert abs(out_put.V.item()) <= 1e-12

    # 5. Put-Call parity holds
    assert abs(price_call - out_put.V.item() - expected_call) <= 1e-12

    # 6. Bitwise equality to analytic
    assert torch.equal(out_call.V, out_ana_call.V)


def test_adversarial_injected_butterfly_arbitrage() -> None:
    """Injected Butterfly Arbitrage Detection.

    Verifies cross_slice_audit flags ok=False on butterfly / slope violating slices.
    """
    engine = AnalyticTriangulated()
    X = _make_clean_test_state()

    # Inject extreme SVI slope b=2.2, rho_s=0.5 -> L1 = 3.3 > 2.0
    X[0, Col.SVI1 + 1] = 2.2
    X[0, Col.SVI1 + 2] = 0.5

    audit_res = cross_slice_audit(engine, X, n_grid=401, z_max=12.0)
    assert audit_res.ok[0].item() is False, "cross_slice_audit failed to catch injected butterfly violation"


def test_adversarial_calendar_arbitrage_detection_gap() -> None:
    """Adversarial Challenge: Probing Calendar Arbitrage Detection in cross_slice_audit.

    Constructs two slices for the same cross where expiry increases from T1=0.2 to T2=1.0,
    but total variance w decreases from 0.05 to 0.01 (violating calendar monotonicity d(w)/dT >= 0).
    Demonstrates whether cross_slice_audit catches cross-tenor calendar arbitrage.
    """
    engine = AnalyticTriangulated()
    state1 = _make_clean_test_state().squeeze(0)
    state2 = _make_clean_test_state().squeeze(0)

    # State 1: T1 = 0.2, vol = 0.50 -> w1 approx 0.05
    state1[Col.T] = 0.2
    state1[Col.T_SET] = 0.2
    state1[Col.SVI1] = (0.50 ** 2) * 0.2
    state1[Col.SVI1 + 4] = 0.10
    state1[Col.SVI2] = (0.50 ** 2) * 0.2
    state1[Col.SVI2 + 4] = 0.10

    # State 2: T2 = 1.0, vol = 0.10 -> w2 approx 0.01 (total variance decreases across maturities!)
    state2[Col.T] = 1.0
    state2[Col.T_SET] = 1.0
    state2[Col.SVI1] = (0.10 ** 2) * 1.0
    state2[Col.SVI1 + 4] = 0.10
    state2[Col.SVI2] = (0.10 ** 2) * 1.0
    state2[Col.SVI2 + 4] = 0.10

    X_pair = torch.stack([state1, state2], dim=0)
    audit_res = cross_slice_audit(engine, X_pair, n_grid=401)

    # CRITICAL AUDIT FINDING:
    # cross_slice_audit evaluates each slice in isolation and fails to detect calendar arbitrage between maturities.
    # We record this empirical observation:
    calendar_violation_caught = not audit_res.ok.all().item()
    if not calendar_violation_caught:
        pytest.fail(
            "DEFECT CONFIRMED: cross_slice_audit returned ok=[True, True] for a multi-expiry batch "
            "with decreasing total variance (w1=0.05 at T=0.2 vs w2=0.01 at T=1.0), failing to enforce "
            "calendar monotonicity across expiries!"
        )


def test_adversarial_output_audit_guardian_routing_gap() -> None:
    """Adversarial Challenge: Probing OUTPUT_AUDIT_FAILED routing in FXCrossGuardian.

    The specification (§1.9, Task 3.2(c)) and worker handoff state:
    'Injected calendar arbitrage slices successfully detected with ok=False and routed with OUTPUT_AUDIT_FAILED.'
    Verifies whether FXCrossGuardian connects to cross_slice_audit and emits OUTPUT_AUDIT_FAILED.
    """
    guardian = FXCrossGuardian()
    X = _make_clean_test_state()

    # Inject an arbitrageable slice
    X[0, Col.SVI1 + 1] = 2.2
    X[0, Col.SVI1 + 2] = 0.5

    decision = guardian.route(X, engine=PricingEngineEnum.DIFFERENTIAL_ML)

    # Check if guardian checked slice audit or routed with OUTPUT_AUDIT_FAILED
    has_audit_failover = any(r == FallbackReasonEnum.OUTPUT_AUDIT_FAILED.value for r in decision.fallback_reason)
    if not has_audit_failover:
        pytest.fail(
            "DEFECT CONFIRMED: FXCrossGuardian.route() does not evaluate cross_slice_audit "
            "and never emits FallbackReasonEnum.OUTPUT_AUDIT_FAILED when given an arbitrageable slice!"
        )
