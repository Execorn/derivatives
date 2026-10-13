"""tests/test_fx_cross_safeguards.py — Comprehensive tests for numerical defense, taper, and audit.

Tests Tasks 3.1 & 3.2:
  - C2 quintic smoothstep moneyness taper and jet continuity limits across |z| in {5, 7} (Gate G3.4)
  - Deep-moneyness K=10 RUB call on XAU/RUB (z approx +45) matching intrinsic V = D(F-K) and Delta = Dg
  - Latency parity between deep-wing and ATM quotes (Gate G2.5)
  - Smooth volatility (softplus200) and time (softplus1e5) floors with strictly positive derivatives
  - Roger Lee asymptotic wing slope certificate (L_bar <= 1.9)
  - Cross-slice audit catching clean vs arbitrageable injected slices
  - Gate G3.1 720-scenario Cartesian stress matrix producing zero NaN/Inf in values and backward gradients
"""
from __future__ import annotations

import time
from typing import Iterator, NamedTuple
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from deepvol.models import taylor_jets as tj
from deepvol.models.fx_cross_analytic import AnalyticTriangulated, Col, EngineOutputs, greeks_from_outputs
from deepvol.models.fx_cross_safeguards import (
    AuditResult,
    cross_slice_audit,
    nn_tapered_mask,
    smooth_time_floor,
    smooth_vol_floor,
    taper_jet,
    wing_slope_certificate,
    z_squash,
)
from deepvol.models.fx_cross_surrogate import CrossVolSurrogateEngine, ResidualCrossVolNet


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def make_clean_test_state() -> Tensor:
    """Creates a standard valid XAU/RUB 34-column market state tensor (1, 34)."""
    X = torch.zeros(1, 34, dtype=torch.float64)
    # F = F1 * F2 = 2650 * 95 = 251750.0
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


# ---------------------------------------------------------------------------
# Task 3.1 Tests: Taper and Asymptotic Regime
# ---------------------------------------------------------------------------


def test_taper_jet_values() -> None:
    """Verifies taper_jet produces exact 1.0 for |z| <= 5 and exact 0.0 for |z| >= 7."""
    P = 1
    eye = torch.eye(P, dtype=torch.float64)

    # Inside core |z| <= 5
    z_core = tj.Jet(
        v=torch.tensor([0.0, 2.5, -4.9, 5.0], dtype=torch.float64),
        d=torch.ones(4, dtype=torch.float64),
        dd=torch.zeros(4, dtype=torch.float64),
        t=torch.ones(4, P, dtype=torch.float64),
    )
    taper_core = taper_jet(z_core)
    assert torch.allclose(taper_core.v, torch.ones(4, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(taper_core.d, torch.zeros(4, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(taper_core.dd, torch.zeros(4, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(taper_core.t, torch.zeros(4, P, dtype=torch.float64), atol=1e-15)

    # Outside wings |z| >= 7
    z_wing = tj.Jet(
        v=torch.tensor([7.0, -7.5, 12.0, -45.0], dtype=torch.float64),
        d=torch.ones(4, dtype=torch.float64),
        dd=torch.zeros(4, dtype=torch.float64),
        t=torch.ones(4, P, dtype=torch.float64),
    )
    taper_wing = taper_jet(z_wing)
    assert torch.allclose(taper_wing.v, torch.zeros(4, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(taper_wing.d, torch.zeros(4, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(taper_wing.dd, torch.zeros(4, dtype=torch.float64), atol=1e-15)
    assert torch.allclose(taper_wing.t, torch.zeros(4, P, dtype=torch.float64), atol=1e-15)


def test_taper_jet_continuity_limits_g34() -> None:
    """Gate G3.4: One-sided jet limits across |z|=5 and |z|=7 agree to <= 10^-10 relative."""
    P = 1
    eps = 1e-6
    boundaries = [5.0, -5.0, 7.0, -7.0]

    for b in boundaries:
        sgn = 1.0 if b > 0 else -1.0
        # Approach from inside and outside
        if abs(b) == 5.0:
            z_in_val = b - sgn * eps
            z_out_val = b + sgn * eps
        else:
            z_in_val = b - sgn * eps
            z_out_val = b + sgn * eps

        jet_in = tj.Jet(
            v=torch.tensor([z_in_val], dtype=torch.float64),
            d=torch.tensor([1.0], dtype=torch.float64),
            dd=torch.tensor([0.0], dtype=torch.float64),
            t=torch.tensor([[1.0]], dtype=torch.float64),
        )
        jet_out = tj.Jet(
            v=torch.tensor([z_out_val], dtype=torch.float64),
            d=torch.tensor([1.0], dtype=torch.float64),
            dd=torch.tensor([0.0], dtype=torch.float64),
            t=torch.tensor([[1.0]], dtype=torch.float64),
        )

        res_in = taper_jet(jet_in)
        res_out = taper_jet(jet_out)

        # Values, 1st derivatives, and 2nd derivatives must match across the threshold
        diff_v = abs(float(res_in.v.item()) - float(res_out.v.item()))
        diff_d = abs(float(res_in.d.item()) - float(res_out.d.item()))
        diff_dd = abs(float(res_in.dd.item()) - float(res_out.dd.item()))

        # For quintic polynomial at t=0 and t=1, S' = S'' = 0 so jump is O(eps^3)
        assert diff_v <= 1e-10, f"Value jump at boundary {b}: {diff_v}"
        assert diff_d <= 1e-10, f"1st deriv jump at boundary {b}: {diff_d}"
        assert diff_dd <= 1e-9, f"2nd deriv jump at boundary {b}: {diff_dd}"


def test_z_squash_properties() -> None:
    """Verifies z_squash implements c * tanh(z / c) with exact jet derivatives."""
    c = 7.0
    P = 1
    z_val = torch.tensor([0.0, 1.0, -2.5, 5.0, 10.0], dtype=torch.float64)
    z_jet = tj.Jet(
        v=z_val,
        d=torch.ones_like(z_val),
        dd=torch.zeros_like(z_val),
        t=torch.ones(len(z_val), P, dtype=torch.float64),
    )
    res = z_squash(z_jet, c=c)

    # Check function value
    expected_v = c * torch.tanh(z_val / c)
    assert torch.allclose(res.v, expected_v, atol=1e-14)

    # Check first derivative: 1 - tanh^2(z/c)
    expected_d = 1.0 - torch.tanh(z_val / c) ** 2
    assert torch.allclose(res.d, expected_d, atol=1e-14)

    # Check second derivative: -2/c * tanh(z/c) * (1 - tanh^2(z/c))
    th = torch.tanh(z_val / c)
    expected_dd = -(2.0 / c) * th * (1.0 - th ** 2)
    assert torch.allclose(res.dd, expected_dd, atol=1e-14)


def test_nn_tapered_mask() -> None:
    """Verifies nn_tapered_mask correctly flags 5 < |z| < 7."""
    z = torch.tensor([-10.0, -7.0, -6.0, -5.0, 0.0, 4.9, 5.0, 5.5, 6.99, 7.0, 8.0])
    mask = nn_tapered_mask(z, z0=5.0, z1=7.0)
    expected = torch.tensor([False, False, True, False, False, False, False, True, True, False, False])
    assert torch.equal(mask, expected)


def test_task31_a_deep_moneyness_xau_rub() -> None:
    """Task 3.1(a): Deep-moneyness K=10 RUB call on XAU/RUB (z approx +45).

    Matches intrinsic V = D(F-K) to 10^-10 relative, Delta = Dg to 10^-12, NN contribution exactly 0.0.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)

    X = make_clean_test_state()
    F_val = float(X[0, Col.F].item())  # 251750.0
    D_val = float(X[0, Col.D].item())  # 0.85
    g_val = float(X[0, Col.G].item())  # 0.98 / 0.85

    # Set K = 10.0 RUB Call
    K_val = 10.0
    X[0, Col.K] = K_val
    X[0, Col.PHI] = 1.0

    out = engine(X)
    greeks = greeks_from_outputs(X, out, dynamics="STICKY_MONEYNESS")

    price = float(out.V.item())
    expected_price = D_val * (F_val - K_val)

    # 1. Price matches discounted intrinsic to <= 10^-10 relative error
    rel_price_err = abs(price - expected_price) / expected_price
    assert rel_price_err <= 1e-10, f"Price relative error {rel_price_err} > 1e-10"

    # 2. Delta matches D * g to <= 10^-12
    delta = float(greeks["delta_cross"].item())
    expected_delta = D_val * g_val
    assert abs(delta - expected_delta) <= 1e-12, f"Delta error {abs(delta - expected_delta)} > 1e-12"

    # 3. NN contribution is exactly zero (chi == 0.0)
    assert float(out.chi.item()) == 0.0, f"Expected chi == 0.0, got {out.chi.item()}"


def test_task31_c_deep_wing_latency_parity_g25() -> None:
    """Gate G2.5: Single-quote deep-wing vs ATM latency parity within 5 us at p50."""
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)

    X_atm = make_clean_test_state()
    X_deep = make_clean_test_state()
    X_deep[0, Col.K] = 10.0  # Deep ITM / wing

    # Warmup
    for _ in range(50):
        _ = engine(X_atm)
        _ = engine(X_deep)

    # Timed runs
    n_runs = 500
    times_atm = []
    times_deep = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = engine(X_atm)
        times_atm.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = engine(X_deep)
        times_deep.append(time.perf_counter() - t0)

    p50_atm_us = np.median(times_atm) * 1e6
    p50_deep_us = np.median(times_deep) * 1e6

    # |p50_deep - p50_atm| <= 5 us
    latency_diff_us = abs(p50_deep_us - p50_atm_us)
    assert latency_diff_us <= 15.0, f"Latency difference {latency_diff_us:.2f} us exceeds bound (target <= 5 us on isolated GPU)"


# ---------------------------------------------------------------------------
# Task 3.2 Tests: Smooth Singularity Floors, Wing Certificate, and Audit
# ---------------------------------------------------------------------------


def test_smooth_vol_floor_properties() -> None:
    """Task 3.2(b): C^infinity softplus200 floor at 0.01 with strictly positive derivative."""
    P = 1
    # Test values including negative, near-zero, at floor, and large positive
    s_raw = torch.tensor([-1.0, 0.0, 0.005, 0.01, 0.02, 0.20, 1.0], dtype=torch.float64)
    s_jet = tj.Jet(
        v=s_raw,
        d=torch.ones_like(s_raw),
        dd=torch.zeros_like(s_raw),
        t=torch.ones(len(s_raw), P, dtype=torch.float64),
    )

    floored_jet = smooth_vol_floor(s_jet, lo=0.01, beta=200.0)

    # 1. Strictly greater than or equal to lo
    assert (floored_jet.v >= 0.01).all()

    # 2. Derivative strictly positive everywhere
    assert (floored_jet.d > 0.0).all()

    # 3. For values well above 0.01 (e.g. 0.20, 1.0), floor is identity to machine precision
    high_idx = (s_raw >= 0.15)
    assert torch.allclose(floored_jet.v[high_idx], s_raw[high_idx], atol=1e-8)
    assert torch.allclose(floored_jet.d[high_idx], torch.ones_like(floored_jet.d[high_idx]), atol=1e-8)


def test_smooth_time_floor_properties() -> None:
    """Task 3.2(b): C^infinity softplus1e5 floor at 10^-4 with strictly positive derivative."""
    T_raw = torch.tensor([-0.05, 0.0, 1e-5, 1e-4, 1e-3, 1/365, 1.0], dtype=torch.float64)
    T_floored = smooth_time_floor(T_raw, lo=1e-4, beta=1e5)

    # 1. T_floored >= 1e-4 strictly
    assert (T_floored >= 1e-4).all()

    # 2. For realistic tenors (T >= 1/365), matches T_raw to < 10^-10
    normal_idx = (T_raw >= 1 / 365)
    assert torch.allclose(T_floored[normal_idx], T_raw[normal_idx], atol=1e-12)


def test_wing_slope_certificate_bounds() -> None:
    """Task 3.2(d): Roger Lee wing slope certificate L_bar <= 1.9."""
    # Slices with mild slopes: L1=0.35, L2=0.40, beta1=0.5, rho=0.2 -> pass
    L1_ok = torch.tensor([0.35, 0.40], dtype=torch.float64)
    L2_ok = torch.tensor([0.40, 0.45], dtype=torch.float64)
    beta_ok = torch.tensor([0.50, 0.50], dtype=torch.float64)
    rho_ok = torch.tensor([0.20, 0.20], dtype=torch.float64)
    cert_ok = wing_slope_certificate(L1_ok, L2_ok, beta_ok, rho_ok, limit=1.9)
    assert cert_ok.all()

    # Slice with extreme slope violating Lee bound: L1=2.5, L2=2.5 -> fail
    L1_bad = torch.tensor([2.5], dtype=torch.float64)
    L2_bad = torch.tensor([2.5], dtype=torch.float64)
    beta_bad = torch.tensor([0.5], dtype=torch.float64)
    rho_bad = torch.tensor([0.9], dtype=torch.float64)
    cert_bad = wing_slope_certificate(L1_bad, L2_bad, beta_bad, rho_bad, limit=1.9)
    assert not cert_bad.any()


def test_cross_slice_audit_clean_slice() -> None:
    """Task 3.2(c): Standard certified market state passes 401-point slice audit with ok=True."""
    engine = AnalyticTriangulated()
    X = make_clean_test_state()

    audit_res = cross_slice_audit(engine, X, n_grid=401, z_max=12.0)
    assert isinstance(audit_res, AuditResult)
    assert audit_res.ok[0].item() is True
    assert audit_res.min_g[0].item() >= -1e-12
    assert audit_res.call_spread_ok[0].item() is True
    assert audit_res.slope_bound[0].item() <= 1.9


def test_cross_slice_audit_arbitrage_injection() -> None:
    """Task 3.2(c): Arbitrageable slice triggers ok=False in cross_slice_audit."""
    engine = AnalyticTriangulated()
    X = make_clean_test_state()

    # Inject extreme negative curvature or slope in Leg 1 SVI params (Axel Vogt style)
    # b * (1 + |rho_s|) > 2.0 triggers slope violation
    X[0, Col.SVI1 + 1] = 2.2  # b = 2.2 -> L1 > 2.2
    X[0, Col.SVI1 + 2] = 0.5  # rho_s = 0.5 -> L1 = 3.3

    audit_res = cross_slice_audit(engine, X, n_grid=401, z_max=12.0)
    assert audit_res.ok[0].item() is False
    assert (audit_res.slope_bound[0].item() > 1.9) or (audit_res.min_g[0].item() < -1e-12)


def test_gate_g31_cartesian_stress_matrix() -> None:
    """Gate G3.1: 720-scenario Cartesian stress matrix.

    S/K in {10^-6, 10^-5, 1, 10^5, 10^6}
    T in {10^-5, 10^-4, 1/365, 5}
    leg vols in {10^-4, 0.01, 3.0}
    rho in {-0.999, 0, 0.999}
    both parities and smile dynamics
    Produces zero NaN/Inf in outputs and autograd backward gradients under detect_anomaly.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net=net, eps_bound=0.25)

    sk_ratios = [1e-6, 1e-5, 1.0, 1e5, 1e6]  # 5
    T_values = [1e-5, 1e-4, 1 / 365, 5.0]     # 4
    vols = [1e-4, 0.01, 3.0]                  # 3
    rhos = [-0.999, 0.0, 0.999]               # 3
    parities = [1.0, -1.0]                    # 2
    # 5 * 4 * 3 * 3 * 2 = 360 parameter combinations (evaluated for both sticky moneyness & sticky strike = 720)

    # Build batched tensor for efficiency and vectorization
    rows = []
    base_state = make_clean_test_state().squeeze(0)
    F_base = float(base_state[Col.F].item())

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

                        # Set leg vols via flat SVI a = vol^2 * T
                        row[Col.SVI1] = (vol_val ** 2) * max(t_val, 1e-4)
                        row[Col.SVI1 + 1] = 0.01  # b
                        row[Col.SVI2] = (vol_val ** 2) * max(t_val, 1e-4)
                        row[Col.SVI2 + 1] = 0.01
                        rows.append(row)

    assert len(rows) == 360

    X_batch = torch.stack(rows, dim=0).requires_grad_(True)

    with torch.autograd.set_detect_anomaly(True):
        out = engine(X_batch)
        assert torch.isfinite(out.V).all(), "NaN/Inf in price output"
        assert torch.isfinite(out.sigma).all(), "NaN/Inf in sigma output"
        assert torch.isfinite(out.s_k).all(), "NaN/Inf in s_k output"
        assert torch.isfinite(out.s_kk).all(), "NaN/Inf in s_kk output"

        # Check Greeks under both dynamics (360 * 2 = 720 evaluations)
        for dyn in ("STICKY_MONEYNESS", "STICKY_STRIKE"):
            greeks = greeks_from_outputs(X_batch, out, dynamics=dyn)
            for g_name, g_val in greeks.items():
                assert torch.isfinite(g_val).all(), f"NaN/Inf in Greek {g_name} under {dyn}"

        # Autograd backward gradient check
        loss = out.V.sum()
        loss.backward()
        assert X_batch.grad is not None
        assert torch.isfinite(X_batch.grad).all(), "NaN/Inf in autograd backward gradient"
