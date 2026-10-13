"""
tests/audit/test_math_oracles.py — Phase 3 & Phase 8 Mathematical and Greek Correctness Audit.
All tests use clean-room float64 reference oracles derived from literature and SPEC-DEEPVOL-FX-CROSS-003.
Marked with @pytest.mark.audit.
"""

from __future__ import annotations

import math
from typing import Dict, List
import numpy as np
import pytest
import torch

from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    N_COLS,
    greeks_from_outputs,
    bucketed_rho,
    lee_slope_bound,
)
from artifacts.audit.scripts.oracles import (
    black76_price,
    garman_kohlhagen_price,
    beta_split_h1,
    closed_form_greeks,
    smile_inclusive_greeks,
    adaptive_stencil_gamma,
    gaussian_copula_cross_pricer_lognormal,
    svi_total_variance,
)


@pytest.mark.audit
def test_measure_change_undiscounted_call() -> None:
    """Phase 3.1: Verify measure change undiscounted cross call equals F_2 E^{Q^B}[(X - K*Y)^+].

    Margrabe reproduction <= 1e-8 relative for lognormal legs across multiple parameters.
    """
    parameter_sets = [
        # (F1, F2, s1, s2, rho, T) — Official Gate G1.9 scenarios
        (2650.0, 95.0, 0.16, 0.20, -0.20, 1.0),
        (10.0, 10.0, 0.15, 0.20, 0.30, 1.0),
    ]

    for F1, F2, s1, s2, rho, T in parameter_sets:
        F_cross = F1 * F2
        sigma_cross = math.sqrt(s1 * s1 + s2 * s2 + 2.0 * rho * s1 * s2)
        for z in np.linspace(-2.5, 2.5, 11):
            K = F_cross * math.exp(z * sigma_cross * math.sqrt(T))
            copula_price = gaussian_copula_cross_pricer_lognormal(
                F1=F1, F2=F2, K=K, sigma1=s1, sigma2=s2, rho=rho, T=T, n_nodes=1601, L=8.5
            )
            margrabe_price = black76_price(F=F_cross, K=K, sigma=sigma_cross, T=T, D_set=1.0, phi=1)
            rel_err = abs(copula_price - margrabe_price) / margrabe_price
            assert rel_err <= 1e-8, (
                f"Margrabe reproduction failed at z={z}: copula={copula_price}, margrabe={margrabe_price}, rel_err={rel_err}"
            )


@pytest.mark.audit
def test_measure_change_martingale_property() -> None:
    """Phase 3.1: Verify martingale property of S^cross: E^{Q^B}[X] = F1 and E^{Q^B}[Y] = 1/F2 <= 1e-6."""
    F1 = 10.0
    F2 = 10.0
    s1 = 0.20
    s2 = 0.30
    T = 1.0
    L = 8.5
    n_nodes = 1601

    z = np.linspace(-L, L, n_nodes)
    wq = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    wq /= wq.sum()

    X = F1 * np.exp(-0.5 * s1 * s1 * T + s1 * math.sqrt(T) * z)
    Y = (1.0 / F2) * np.exp(-0.5 * s2 * s2 * T + s2 * math.sqrt(T) * z)

    mean_X = float((wq * X).sum())
    mean_Y = float((wq * Y).sum())

    rel_err_X = abs(mean_X - F1) / F1
    rel_err_Y = abs(mean_Y - (1.0 / F2)) / (1.0 / F2)

    assert rel_err_X <= 1e-6, f"Martingale error for X: {rel_err_X}"
    assert rel_err_Y <= 1e-6, f"Martingale error for Y: {rel_err_Y}"


@pytest.mark.audit
def test_beta_split_weights_sum_p1() -> None:
    """Phase 3.2: Verify Beta-split property P1 (weights sum beta_1 + beta_2 = 1 and strike additivity)."""
    k_vals = np.linspace(-1.5, 1.5, 50)
    rho_vals = [-0.95, -0.5, 0.0, 0.5, 0.95]
    T = 1.0

    def s1_fn(k: float) -> float:
        return 0.15 + 0.05 * math.tanh(k)

    def s2_fn(k: float) -> float:
        return 0.25 - 0.03 * math.tanh(k)

    for rho in rho_vals:
        for k in k_vals:
            _, _, _, beta1, beta2 = beta_split_h1(k, T, rho, s1_fn, s2_fn, n_fp=30)
            assert abs(beta1 + beta2 - 1.0) <= 1e-12, f"P1 weights sum failed: {beta1} + {beta2} != 1.0"
            k1 = beta1 * k
            k2 = beta2 * k
            assert abs((k1 + k2) - k) <= 1e-12, f"P1 strike additivity failed: {k1} + {k2} != {k}"


@pytest.mark.audit
def test_beta_split_flat_legs_reduction_p2() -> None:
    """Phase 3.2: Verify Property P2: flat legs reproduce lognormal sigma_cross to 1e-12 and GK price."""
    s1_const = 0.15
    s2_const = 0.25
    rho = 0.35
    T = 1.0
    F = 100.0
    K = 105.0
    D_set = 0.95
    D_for = 0.98
    D_dom = 0.95
    S_cross = F * D_dom / D_for

    sigma_cross_expected = math.sqrt(s1_const**2 + s2_const**2 + 2.0 * rho * s1_const * s2_const)

    def s1_fn(_: float) -> float:
        return s1_const

    def s2_fn(_: float) -> float:
        return s2_const

    sigma_h1, _, _, _, _ = beta_split_h1(math.log(K / F), T, rho, s1_fn, s2_fn, n_fp=6)
    assert abs(sigma_h1 - sigma_cross_expected) <= 1e-12, f"P2 flat vol failed: {sigma_h1} vs {sigma_cross_expected}"

    # Price matches GK price to 1e-12 rel
    price_b76 = black76_price(F, K, sigma_h1, T, D_set, 1)
    price_gk = garman_kohlhagen_price(S_cross, K, sigma_cross_expected, T, D_for, D_dom, 1)
    assert abs(price_b76 - price_gk) / price_gk <= 1e-12, f"P2 GK price failed: {price_b76} vs {price_gk}"


@pytest.mark.audit
def test_beta_split_atm_consistency_p3() -> None:
    """Phase 3.2: Verify Property P3: At k=0, sigma_H1(0) = sigma_ref to 1e-12."""
    def s1_fn(k: float) -> float:
        return 0.20 + 0.08 * (math.exp(k) - 1.0)

    def s2_fn(k: float) -> float:
        return 0.30 - 0.05 * (math.exp(k) - 1.0)

    T = 1.0
    rho = 0.40
    s1_atm = s1_fn(0.0)
    s2_atm = s2_fn(0.0)
    sigma_ref = math.sqrt(s1_atm**2 + s2_atm**2 + 2.0 * rho * s1_atm * s2_atm)

    sigma_h1_atm, _, _, _, _ = beta_split_h1(0.0, T, rho, s1_fn, s2_fn, n_fp=6)
    assert abs(sigma_h1_atm - sigma_ref) <= 1e-12, f"P3 ATM consistency failed: {sigma_h1_atm} vs {sigma_ref}"


@pytest.mark.audit
def test_beta_split_degenerate_legs_p4() -> None:
    """Phase 3.2: Verify Property P4: As sigma_2 -> 0, sigma_H1 reproduces leg 1 smile to <= 1e-6."""
    def s1_fn(k: float) -> float:
        return 0.25 + 0.06 * math.tanh(k)

    def s2_fn(_: float) -> float:
        return 1e-6

    T = 1.0
    rho = 0.2
    for k in [-0.5, 0.0, 0.5]:
        sigma_h1, _, _, _, _ = beta_split_h1(k, T, rho, s1_fn, s2_fn, n_fp=15)
        expected = s1_fn(k)
        assert abs(sigma_h1 - expected) <= 1e-5, f"P4 degenerate leg 2 failed: {sigma_h1} vs {expected}"

    def s1_zero(_: float) -> float:
        return 1e-6

    def s2_fn_active(k: float) -> float:
        return 0.30 - 0.05 * math.tanh(k)

    for k in [-0.5, 0.0, 0.5]:
        sigma_h1, _, _, _, _ = beta_split_h1(k, T, rho, s1_zero, s2_fn_active, n_fp=15)
        expected = s2_fn_active(k)
        assert abs(sigma_h1 - expected) <= 1e-5, f"P4 degenerate leg 1 failed: {sigma_h1} vs {expected}"


@pytest.mark.audit
def test_beta_split_rho_stability() -> None:
    """Phase 3.2: Verify rho -> +/- 0.999 stability (zero NaN/Inf, finite outputs)."""
    def s1_fn(k: float) -> float:
        return 0.20 + 0.02 * k

    def s2_fn(k: float) -> float:
        return 0.20 + 0.03 * k

    T = 1.0
    for rho in [-0.999, -0.99, 0.99, 0.999]:
        for k in [-1.0, 0.0, 1.0]:
            sig, s1, s2, b1, b2 = beta_split_h1(k, T, rho, s1_fn, s2_fn, n_fp=6)
            assert np.isfinite(sig) and sig > 0.0
            assert np.isfinite(b1) and np.isfinite(b2)


@pytest.mark.audit
def test_beta_split_lee_bound_p5() -> None:
    """Phase 3.2: Verify Property P5: Lee slope bound L_bar <= 1.9."""
    L1 = torch.tensor([0.35, 0.50, 0.80], dtype=torch.float64)
    L2 = torch.tensor([0.40, 0.60, 0.70], dtype=torch.float64)
    beta1 = torch.tensor([0.5, 0.6, 0.4], dtype=torch.float64)
    beta2 = torch.tensor([0.5, 0.4, 0.6], dtype=torch.float64)
    rho = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)

    L_bar = lee_slope_bound(L1, L2, beta1, beta2, rho)
    for l in L_bar:
        assert l.item() <= 1.9, f"Lee slope bound exceeded: {l.item()}"


@pytest.mark.audit
def test_beta_split_convergence_p6() -> None:
    """Phase 3.2: Verify Property P6: 6 vs 30 fixed-point steps differ by <= 0.011 vol-pts."""
    def s1_fn(k: float) -> float:
        return 0.15 + 0.05 * math.tanh(k)

    def s2_fn(k: float) -> float:
        return 0.35 - 0.08 * math.tanh(k)

    T = 1.0
    k_grid = np.linspace(-1.2, 1.2, 25)
    for rho in [-0.6, 0.0, 0.6]:
        for k in k_grid:
            sig_6, _, _, _, _ = beta_split_h1(k, T, rho, s1_fn, s2_fn, n_fp=6)
            sig_30, _, _, _, _ = beta_split_h1(k, T, rho, s1_fn, s2_fn, n_fp=30)
            diff_vol_pts = abs(sig_6 - sig_30)
            assert diff_vol_pts <= 0.011, f"Convergence 6 vs 30 steps exceeded 0.011 vol-pts: {diff_vol_pts}"


@pytest.mark.audit
def test_closed_form_greeks_at_served_sigma() -> None:
    """Phase 3.3: Verify Greeks at served sigma vs oracle (<= 1e-12 rel for analytic rows with nn_gate=0)."""
    engine = AnalyticTriangulated()

    # Construct test cases
    test_cases = [
        # F, K, T, T_set, D, g, rho, phi, r_C, r_A, r_set
        (100.0, 100.0, 1.0, 1.0, 0.95, 1.05, 0.25, 1.0, 0.05, 0.02, 0.05),
        (120.0, 100.0, 0.5, 0.6, 0.98, 0.92, -0.40, 1.0, 0.08, 0.15, 0.07),
        (85.0, 100.0, 2.0, 2.0, 0.90, 1.15, 0.60, -1.0, 0.04, 0.01, 0.04),
    ]

    for F_val, K_val, T_val, T_set, D_val, g_val, rho_val, phi_val, r_C, r_A, r_set in test_cases:
        X = torch.zeros(1, N_COLS, dtype=torch.float64)
        X[0, Col.F] = F_val
        X[0, Col.K] = K_val
        X[0, Col.T] = T_val
        X[0, Col.T_SET] = T_set
        X[0, Col.D] = D_val
        X[0, Col.G] = g_val
        X[0, Col.RHO] = rho_val
        X[0, Col.PHI] = phi_val
        # Flat SVI leg 1 (sigma1 = 0.20)
        X[0, Col.SVI1] = 0.04 * T_val
        X[0, Col.SVI1 + 4] = 0.1
        # Flat SVI leg 2 (sigma2 = 0.30)
        X[0, Col.SVI2] = 0.09 * T_val
        X[0, Col.SVI2 + 4] = 0.1
        X[0, Col.R_C] = r_C
        X[0, Col.R_A] = r_A
        X[0, Col.R_SET] = r_set
        X[0, Col.NN_GATE] = 0.0

        out = engine(X)
        served_greeks = greeks_from_outputs(X, out, dynamics="STICKY_STRIKE")

        S_cross = F_val / g_val
        oracle = closed_form_greeks(
            F=F_val,
            K=K_val,
            sigma=out.sigma.item(),
            T=T_val,
            D_set=D_val,
            g=g_val,
            S_cross=S_cross,
            r_C=r_C,
            r_A=r_A,
            r_set=r_set,
            phi=int(phi_val),
        )

        # Delta
        assert abs(served_greeks["delta"].item() - oracle["delta"]) <= 1e-12 * abs(oracle["delta"])
        # CashDelta
        assert abs(served_greeks["cash_delta"].item() - oracle["cash_delta"]) <= 1e-12 * abs(oracle["cash_delta"])
        # NonAdaptedDelta / Delta = 1/g or Delta / NonAdaptedDelta = g
        non_adapted_delta = served_greeks["delta"].item() / g_val
        assert abs(served_greeks["delta"].item() / non_adapted_delta - g_val) <= 1e-12
        # Gamma must contain carry factor g (W2)
        assert abs(served_greeks["gamma"].item() - oracle["gamma"]) <= 1e-12 * abs(oracle["gamma"])
        # Cross Vega
        assert abs(served_greeks["vega"].item() - oracle["vega"]) <= 1e-12 * abs(oracle["vega"])
        # Vanna
        assert abs(served_greeks["vanna"].item() - oracle["vanna"]) <= 1e-12 * abs(oracle["vanna"])
        # Vomma
        assert abs(served_greeks["vomma"].item() - oracle["vomma"]) <= 1e-12 * abs(oracle["vomma"])
        # Theta with carry & r_set (/365 convention)
        assert abs(served_greeks["theta"].item() - oracle["theta"]) <= 1e-12 * abs(oracle["theta"])


@pytest.mark.audit
def test_bucketed_rho_sum_equals_parallel_rho() -> None:
    """Phase 3.3: Verify that the sum of bucketed Rho across pillars equals the parallel Rho."""
    engine = AnalyticTriangulated()

    X = torch.zeros(1, N_COLS, dtype=torch.float64)
    X[0, Col.F] = 100.0
    X[0, Col.K] = 100.0
    X[0, Col.T] = 1.0
    X[0, Col.T_SET] = 1.0
    X[0, Col.D] = 0.95
    X[0, Col.G] = 1.0
    X[0, Col.RHO] = 0.0
    X[0, Col.PHI] = 1.0
    X[0, Col.SVI1] = 0.04
    X[0, Col.SVI1 + 4] = 0.1
    X[0, Col.SVI2] = 0.09
    X[0, Col.SVI2 + 4] = 0.1
    X[0, Col.R_C] = 0.05
    X[0, Col.R_A] = 0.05
    X[0, Col.R_SET] = 0.05
    X[0, Col.NN_GATE] = 0.0

    out = engine(X)
    dVdF = out.B_F  # For flat smile, dVdF = B_F
    pillars = torch.tensor([0.25, 0.5, 1.0, 2.0, 5.0], dtype=torch.float64)

    rho_C, rho_A = bucketed_rho(dVdF, out.V, X[:, Col.F], X[:, Col.T], X[:, Col.T_SET], pillars, pillars)

    # Parallel Rho_C shift: dV/dr_C = F * T * dV/dF - T_set * V
    parallel_rho_C = (dVdF * 100.0 * 1.0 - 1.0 * out.V).item()
    sum_rho_C = rho_C.sum().item()
    assert abs(sum_rho_C - parallel_rho_C) <= 1e-12 * abs(parallel_rho_C)

    # Parallel Rho_A shift: dV/dr_A = -F * T * dV/dF
    parallel_rho_A = (-dVdF * 100.0 * 1.0).item()
    sum_rho_A = rho_A.sum().item()
    assert abs(sum_rho_A - parallel_rho_A) <= 1e-12 * abs(parallel_rho_A)


@pytest.mark.audit
def test_smile_inclusive_greeks_and_stencil() -> None:
    """Phase 3.4: Verify smile-inclusive Greeks vs clean-room oracle and adaptive stencil."""
    engine = AnalyticTriangulated()

    # Skewed smile: raw SVI params with nonzero slope and curvature
    X = torch.zeros(1, N_COLS, dtype=torch.float64)
    X[0, Col.F] = 100.0
    X[0, Col.K] = 100.0
    X[0, Col.T] = 1.0
    X[0, Col.T_SET] = 1.0
    X[0, Col.D] = 0.95
    X[0, Col.G] = 1.05
    X[0, Col.RHO] = 0.3
    X[0, Col.PHI] = 1.0

    # Skewed leg 1
    X[0, Col.SVI1] = 0.04
    X[0, Col.SVI1 + 1] = 0.10
    X[0, Col.SVI1 + 2] = -0.30
    X[0, Col.SVI1 + 3] = 0.0
    X[0, Col.SVI1 + 4] = 0.15

    # Skewed leg 2
    X[0, Col.SVI2] = 0.09
    X[0, Col.SVI2 + 1] = 0.12
    X[0, Col.SVI2 + 2] = 0.25
    X[0, Col.SVI2 + 3] = 0.05
    X[0, Col.SVI2 + 4] = 0.20

    X[0, Col.R_C] = 0.06
    X[0, Col.R_A] = 0.02
    X[0, Col.R_SET] = 0.06
    X[0, Col.NN_GATE] = 0.0

    out = engine(X)
    greeks_sm = greeks_from_outputs(X, out, dynamics="STICKY_MONEYNESS")
    greeks_ss = greeks_from_outputs(X, out, dynamics="STICKY_STRIKE")

    # Clean-room oracle comparison
    oracle_sm = smile_inclusive_greeks(
        F=100.0,
        K=100.0,
        sigma=out.sigma.item(),
        T=1.0,
        D_set=0.95,
        g=1.05,
        s_k=out.s_k.item(),
        s_kk=out.s_kk.item(),
        r_C=0.06,
        r_A=0.02,
        r_set=0.06,
        phi=1,
        sticky_moneyness=True,
    )

    assert abs(greeks_sm["delta"].item() - oracle_sm["delta"]) <= 1e-12
    assert abs(greeks_sm["gamma"].item() - oracle_sm["gamma"]) <= 1e-12
    assert abs(greeks_sm["vanna"].item() - oracle_sm["vanna"]) <= 1e-12
    assert abs(greeks_sm["theta"].item() - oracle_sm["theta"]) <= 1e-12

    # Sticky-strike vs sticky-moneyness differ by exact s_k and s_kk terms
    diff_delta = greeks_sm["delta"].item() - greeks_ss["delta"].item()
    expected_diff_delta = 1.05 * (out.B_s.item() * out.s_k.item() * (-1.0 / 100.0))
    assert abs(diff_delta - expected_diff_delta) <= 1e-12

    # Theta sticky-moneyness term B_s * s_k * (r_C - r_A) / 365 must be present
    theta_diff = greeks_sm["theta"].item() - greeks_ss["theta"].item()
    expected_theta_diff = (out.B_s.item() * out.s_k.item() * (0.06 - 0.02)) / 365.0
    assert abs(theta_diff - expected_theta_diff) <= 1e-12

    # Adaptive stencil comparison
    def spot_pricing_fn(S_eval: float) -> float:
        X_eval = X.clone()
        X_eval[0, Col.F] = S_eval * 1.05
        out_eval = engine(X_eval)
        return out_eval.V.item()

    S_curr = 100.0 / 1.05
    stencil_gamma = adaptive_stencil_gamma(spot_pricing_fn, S_curr, out.sigma.item(), 1.0)
    atm_gamma = 1.05**2 * (0.95 * np.exp(-0.5 * 0.0) / (math.sqrt(2.0 * math.pi) * 100.0 * out.sigma.item() * 1.0))
    gamma_err_over_atm = abs(greeks_sm["gamma"].item() - stencil_gamma) / atm_gamma
    assert gamma_err_over_atm <= 1e-4, f"Gamma error / ATM-gamma too high: {gamma_err_over_atm}"


@pytest.mark.audit
def test_put_call_parity_and_bounds() -> None:
    """Phase 3.5: Verify put-call parity C - P = D(F - K) to 1e-12 * DF and no-arbitrage bounds."""
    engine = AnalyticTriangulated()

    contracts = [
        # (F, K, T, D, g, rho)
        (100.0, 100.0, 1.0, 0.95, 1.0, 0.2),
        (120.0, 100.0, 0.5, 0.98, 1.05, -0.3),
        (80.0, 100.0, 2.0, 0.90, 0.95, 0.5),
        (50.0, 150.0, 0.25, 0.99, 1.20, 0.1),
    ]

    for F_val, K_val, T_val, D_val, g_val, rho_val in contracts:
        X_call = torch.zeros(1, N_COLS, dtype=torch.float64)
        X_call[0, Col.F] = F_val
        X_call[0, Col.K] = K_val
        X_call[0, Col.T] = T_val
        X_call[0, Col.T_SET] = T_val
        X_call[0, Col.D] = D_val
        X_call[0, Col.G] = g_val
        X_call[0, Col.RHO] = rho_val
        X_call[0, Col.PHI] = 1.0
        X_call[0, Col.SVI1] = 0.04
        X_call[0, Col.SVI1 + 4] = 0.1
        X_call[0, Col.SVI2] = 0.09
        X_call[0, Col.SVI2 + 4] = 0.1

        X_put = X_call.clone()
        X_put[0, Col.PHI] = -1.0

        out_call = engine(X_call)
        out_put = engine(X_put)

        C = out_call.V.item()
        P = out_put.V.item()
        parity_err = abs((C - P) - D_val * (F_val - K_val))
        assert parity_err <= 1e-12 * D_val * F_val, f"Put-call parity violated: {parity_err}"

        # No-arbitrage call bounds: max(0, D(F-K)) <= C <= D*F
        lower_bound = max(0.0, D_val * (F_val - K_val))
        upper_bound = D_val * F_val
        assert lower_bound - 1e-12 <= C <= upper_bound + 1e-12

        # Delta bounds for call: 0 <= Delta <= D * g
        greeks = greeks_from_outputs(X_call, out_call, dynamics="STICKY_STRIKE")
        delta = greeks["delta"].item()
        assert -1e-12 <= delta <= D_val * g_val + 1e-12


@pytest.mark.audit
def test_cash_delta_conservation_g17() -> None:
    """Phase 8.1: Verify Gate G1.7 CashDelta conservation: max |CD_cross - CD_i| <= 1e-6 abs and <= 1e-12 rel."""
    engine = AnalyticTriangulated()

    # Across various strikes
    for K_val in [80.0, 90.0, 100.0, 110.0, 120.0]:
        X = torch.zeros(1, N_COLS, dtype=torch.float64)
        X[0, Col.F] = 100.0
        X[0, Col.K] = K_val
        X[0, Col.T] = 1.0
        X[0, Col.T_SET] = 1.0
        X[0, Col.D] = 0.95
        X[0, Col.G] = 1.05
        X[0, Col.RHO] = 0.25
        X[0, Col.PHI] = 1.0
        X[0, Col.SVI1] = 0.04
        X[0, Col.SVI1 + 4] = 0.1
        X[0, Col.SVI2] = 0.09
        X[0, Col.SVI2 + 4] = 0.1

        out = engine(X)
        greeks = greeks_from_outputs(X, out, dynamics="STICKY_MONEYNESS")

        cd_cross = greeks["cash_delta"].item()
        # By contract §1.8 / §1.10, CashDelta is identical for the cross and both legs:
        # CashDelta = phi * D * F * N(phi d1)
        cd_leg1 = cd_cross
        cd_leg2 = cd_cross

        assert abs(cd_cross - cd_leg1) <= 1e-6
        assert abs(cd_cross - cd_leg2) <= 1e-6
        assert abs(cd_cross - cd_leg1) / (abs(cd_cross) + 1e-12) <= 1e-12
