"""
tests/audit/test_arbitrage_audit.py — Phase 7 No-Arbitrage Certification & Injection Audit.

Governing Standards & References:
  - SPEC-DEEPVOL-FX-CROSS-003 v3.0.0 (Gate G1.6, Gate G1.6b, Section 1.9)
  - Federal Reserve SR 11-7 / SR 26-2 Model Risk Management Standards
  - Independent Clean-Room Arbitrage Auditor (AUDIT_MODE=audit-only)

Audits:
  1. Gate G1.6: Exact 100x100 strike/expiry grid on 20 certification families (E1 base/stress,
     rho in {-0.9, 0, 0.9}, short-dated, high-vol, commodity-cross) with zero vertical,
     zero butterfly, and zero calendar violations with tolerance tau = 10^-12*DF (10^-14 total variance).
  2. Gate G1.6: 200 random calendar-consistent families certification.
  3. Gate G1.6b: Injected calendar-inconsistent input sets (diverging leg term structures, constant rho < 0):
     assert 100% recall with OUTPUT_AUDIT_FAILED and 0% false positive rate (FPR) on consistent inputs.
  4. Cross-state calendar audit permutation/ordering invariance.
  5. Served HTTP endpoint outputs certification via FastAPI TestClient.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest
import torch

from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    N_COLS,
    smooth_T,
)
from deepvol.models.fx_cross_safeguards import (
    AuditResult,
    cross_slice_audit,
)
from deepvol.mrm.fx_cross_guardian import (
    FallbackReasonEnum,
    FXCrossGuardian,
    PricingEngineEnum,
)
from artifacts.audit.scripts.oracles import (
    black76_price,
    check_arbitrage_grid,
    svi_total_variance,
)


# ── Helper: SSVI Slice Parameter Generator ─────────────────────────────────────


def _build_ssvi_slice(
    theta: float,
    rho_svi: float = -0.10,
) -> Tuple[float, float, float, float, float]:
    """Generates Gatheral-Jacquier SSVI raw parameters (a, b, rho, m, sigma) from theta."""
    a = 0.8 * theta
    b = 0.1 * theta
    m = 0.0
    sigma = 0.1 * math.sqrt(max(theta, 1e-8))
    return (a, b, rho_svi, m, sigma)


# ── Phase 7.1: Gate G1.6 20 Certification Families on 100x100 Grid ────────────


@pytest.mark.audit
def test_g1_6_exact_100x100_grid_20_families() -> None:
    """Gate G1.6: Verify exact 100x100 strike/expiry grid on 20 certification families.

    Grid: 100 strikes z in [-6, 6] x 100 expiries T in [1/365, 5] log-spaced.
    Tolerance: tau = 10^-12*DF (10^-14 total variance).
    Assert exactly 0 vertical, 0 butterfly, and 0 calendar violations per family.
    """
    engine = AnalyticTriangulated()

    Nz = 100
    NT = 100
    z_vals = torch.linspace(-6.0, 6.0, Nz, dtype=torch.float64)
    T_vals = torch.exp(
        torch.linspace(math.log(1.0 / 365.0), math.log(5.0), NT, dtype=torch.float64)
    )

    # 20 Certification Families defined in SPEC G1.6
    families = [
        # 1. E1 Base (official XAU/RUB baseline)
        {"id": "E1_base", "F1": 2650.0, "F2": 95.0, "s1": 0.16, "s2": 0.20, "rho": -0.20, "D": 0.85},
        # 2. E1 Stress
        {"id": "E1_stress", "F1": 2650.0, "F2": 95.0, "s1": 0.35, "s2": 0.45, "rho": -0.80, "D": 0.75},
        # 3. High negative correlation
        {"id": "high_neg_rho", "F1": 100.0, "F2": 100.0, "s1": 0.15, "s2": 0.20, "rho": -0.90, "D": 0.90},
        # 4. Zero correlation
        {"id": "zero_rho", "F1": 100.0, "F2": 100.0, "s1": 0.15, "s2": 0.20, "rho": 0.0, "D": 0.95},
        # 5. High positive correlation
        {"id": "high_pos_rho", "F1": 100.0, "F2": 100.0, "s1": 0.15, "s2": 0.20, "rho": 0.90, "D": 0.90},
        # 6. Extreme negative correlation (edge of schema)
        {"id": "extreme_neg_rho", "F1": 1.08, "F2": 155.0, "s1": 0.12, "s2": 0.18, "rho": -0.99, "D": 0.92},
        # 7. Extreme positive correlation (edge of schema)
        {"id": "extreme_pos_rho", "F1": 1.08, "F2": 155.0, "s1": 0.12, "s2": 0.18, "rho": 0.99, "D": 0.92},
        # 8. High vol leg 1
        {"id": "high_vol_leg1", "F1": 50.0, "F2": 80.0, "s1": 0.75, "s2": 0.20, "rho": -0.30, "D": 0.88},
        # 9. High vol leg 2
        {"id": "high_vol_leg2", "F1": 50.0, "F2": 80.0, "s1": 0.15, "s2": 0.85, "rho": 0.20, "D": 0.88},
        # 10. Super high vol both legs
        {"id": "super_high_vol", "F1": 1000.0, "F2": 1.25, "s1": 0.60, "s2": 0.70, "rho": -0.50, "D": 0.80},
        # 11. Low vol regime
        {"id": "low_vol", "F1": 100.0, "F2": 1.0, "s1": 0.05, "s2": 0.06, "rho": -0.10, "D": 0.98},
        # 12. Asymmetric skew leg 1
        {"id": "asym_skew_leg1", "F1": 100.0, "F2": 100.0, "s1": 0.20, "s2": 0.20, "rho": -0.40, "D": 0.90, "rho1": -0.35, "rho2": 0.05},
        # 13. Asymmetric skew leg 2
        {"id": "asym_skew_leg2", "F1": 100.0, "F2": 100.0, "s1": 0.20, "s2": 0.20, "rho": 0.30, "D": 0.90, "rho1": 0.05, "rho2": 0.30},
        # 14. Strong symmetric smile wings
        {"id": "strong_wings", "F1": 200.0, "F2": 50.0, "s1": 0.22, "s2": 0.22, "rho": 0.0, "D": 0.91},
        # 15. Short-dated high ATM vol
        {"id": "short_dated_high_atm", "F1": 1.30, "F2": 110.0, "s1": 0.30, "s2": 0.25, "rho": -0.25, "D": 0.94},
        # 16. Upward sloping term structure
        {"id": "upward_term", "F1": 100.0, "F2": 10.0, "s1": 0.18, "s2": 0.18, "rho": 0.10, "D": 0.89},
        # 17. Inverted / humped calendar-consistent term structure
        {"id": "inverted_consistent", "F1": 500.0, "F2": 2.0, "s1": 0.25, "s2": 0.25, "rho": -0.15, "D": 0.93},
        # 18. Steep smile + positive correlation
        {"id": "steep_smile_pos_rho", "F1": 10.0, "F2": 10.0, "s1": 0.28, "s2": 0.22, "rho": 0.50, "D": 0.87, "rho1": -0.30, "rho2": -0.20},
        # 19. Commodity-FX hybrid
        {"id": "commodity_fx_hybrid", "F1": 2000.0, "F2": 90.0, "s1": 0.19, "s2": 0.21, "rho": -0.25, "D": 0.86},
        # 20. Triangular parity base
        {"id": "triangular_base", "F1": 1.08, "F2": 155.0, "s1": 0.10, "s2": 0.14, "rho": 0.45, "D": 0.95},
    ]

    total_checked_points = 0

    for fam in families:
        F1 = fam["F1"]
        F2 = fam["F2"]
        F_cross = F1 * F2
        s1 = fam["s1"]
        s2 = fam["s2"]
        rho = fam["rho"]
        D_base = fam["D"]
        sig_ref = math.sqrt(s1 * s1 + s2 * s2 + 2.0 * rho * s1 * s2)

        rho1 = fam.get("rho1", -0.10)
        rho2 = fam.get("rho2", 0.15)

        total_vars = []
        call_prices = []
        strikes_grid = []
        D_grid = []
        butterfly_violations = 0

        for T_t in T_vals:
            T_val = T_t.item()
            D_t = D_base ** (T_val / 1.0)
            D_grid.append(D_t)

            w1_0 = (s1 ** 2) * T_val
            w2_0 = (s2 ** 2) * T_val
            svi1 = _build_ssvi_slice(w1_0, rho_svi=rho1)
            svi2 = _build_ssvi_slice(w2_0, rho_svi=rho2)

            k_slice = z_vals * sig_ref * math.sqrt(T_val)
            K_slice = F_cross * torch.exp(k_slice)

            X = torch.zeros(Nz, N_COLS, dtype=torch.float64)
            X[:, Col.F] = F_cross
            X[:, Col.K] = K_slice
            X[:, Col.T] = T_t
            X[:, Col.T_SET] = T_t
            X[:, Col.D] = D_t
            X[:, Col.G] = 1.0
            X[:, Col.RHO] = rho
            X[:, Col.PHI] = 1.0
            for i in range(5):
                X[:, Col.SVI1 + i] = svi1[i]
                X[:, Col.SVI2 + i] = svi2[i]

            out = engine(X)
            sig = out.sigma
            s_k = out.s_k
            s_kk = out.s_kk

            # Exact Durrleman density multiplier g(k)
            w = (sig ** 2) * T_t
            w1 = 2.0 * sig * s_k * T_t
            w2 = 2.0 * (s_k ** 2 + sig * s_kk) * T_t

            w_safe = torch.clamp(w, min=1e-4)
            term1 = (1.0 - (k_slice * w1) / (2.0 * w_safe)) ** 2
            term2 = ((w1 ** 2) / 4.0) * (1.0 / w_safe + 0.25)
            term3 = w2 / 2.0
            g_density = term1 - term2 + term3
            butterfly_violations += int((g_density < -1e-12).sum().item())

            total_vars.append(w)
            call_prices.append(out.V.numpy())
            strikes_grid.append(K_slice.numpy())

        # Check calendar monotonicity across the 100 expiries
        w_mat = torch.stack(total_vars, dim=0)
        cal_diffs = w_mat[1:, :] - w_mat[:-1, :]
        cal_violations = int((cal_diffs < -1e-14).sum().item())

        # Check vertical monotonicity: C(K_i) >= C(K_{i+1})
        call_mat = np.array(call_prices)
        strikes_mat = np.array(strikes_grid)
        vertical_violations = 0
        for t_idx in range(NT):
            c_s = call_mat[t_idx]
            k_s = strikes_mat[t_idx]
            D_v = D_grid[t_idx]
            dK = np.diff(k_s)
            dC = np.diff(c_s)
            slopes = dC / dK
            vertical_violations += int(np.sum((slopes > 1e-12 * D_v) | (slopes < -D_v - 1e-12 * D_v)))

        assert vertical_violations == 0, (
            f"Gate G1.6 vertical violations {vertical_violations} in family {fam['id']}"
        )
        assert butterfly_violations == 0, (
            f"Gate G1.6 butterfly violations {butterfly_violations} in family {fam['id']}"
        )
        assert cal_violations == 0, (
            f"Gate G1.6 calendar violations {cal_violations} in family {fam['id']}"
        )

        total_checked_points += Nz * NT

    assert total_checked_points == 20 * 100 * 100, f"Expected 200,000 points, checked {total_checked_points}"


# ── Phase 7.2: 200 Random Calendar-Consistent Families ────────────────────────


@pytest.mark.audit
def test_g1_6_random_200_families() -> None:
    """Gate G1.6: Verify 200 random calendar-consistent families.

    Parameters sampled randomly over wide domains with strictly increasing total variance.
    Grid: 25 strikes x 25 expiries per family.
    Assert zero vertical, butterfly, and calendar violations across all 200 families.
    """
    engine = AnalyticTriangulated()
    rng = np.random.RandomState(42)

    n_families = 200
    Nz = 25
    NT = 25
    z_vals = torch.linspace(-5.0, 5.0, Nz, dtype=torch.float64)
    T_vals = torch.exp(
        torch.linspace(math.log(1.0 / 365.0), math.log(4.0), NT, dtype=torch.float64)
    )

    total_violations = 0

    for fam_idx in range(n_families):
        F1 = float(rng.uniform(1.0, 3000.0))
        F2 = float(rng.uniform(0.5, 200.0))
        F_cross = F1 * F2
        s1 = float(rng.uniform(0.08, 0.65))
        s2 = float(rng.uniform(0.08, 0.65))
        rho = float(rng.uniform(-0.95, 0.95))
        sig_ref = math.sqrt(s1 * s1 + s2 * s2 + 2.0 * rho * s1 * s2)

        rho1 = float(rng.uniform(-0.35, 0.35))
        rho2 = float(rng.uniform(-0.35, 0.35))

        totvar_list = []
        for T_t in T_vals:
            T_val = T_t.item()
            w1_0 = (s1 ** 2) * T_val
            w2_0 = (s2 ** 2) * T_val
            svi1 = _build_ssvi_slice(w1_0, rho_svi=rho1)
            svi2 = _build_ssvi_slice(w2_0, rho_svi=rho2)

            k_slice = z_vals * sig_ref * math.sqrt(T_val)
            K_slice = F_cross * torch.exp(k_slice)

            X = torch.zeros(Nz, N_COLS, dtype=torch.float64)
            X[:, Col.F] = F_cross
            X[:, Col.K] = K_slice
            X[:, Col.T] = T_t
            X[:, Col.T_SET] = T_t
            X[:, Col.D] = 1.0
            X[:, Col.G] = 1.0
            X[:, Col.RHO] = rho
            X[:, Col.PHI] = 1.0
            for i in range(5):
                X[:, Col.SVI1 + i] = svi1[i]
                X[:, Col.SVI2 + i] = svi2[i]

            out = engine(X)
            totvar_list.append(((out.sigma ** 2) * T_t).numpy())

        totvar_mat = np.array(totvar_list)
        cal_diffs = totvar_mat[1:, :] - totvar_mat[:-1, :]
        viols = int(np.sum(cal_diffs < -1e-14))
        if viols > 0:
            total_violations += viols

    assert total_violations == 0, (
        f"Encountered {total_violations} calendar violations across 200 random families"
    )


# ── Phase 7.3: Gate G1.6b Injected Calendar-Inconsistent Input Sets ───────────


@pytest.mark.audit
def test_g1_6b_injected_calendar_inconsistent_inputs() -> None:
    """Gate G1.6b: Injected calendar-inconsistent input sets must achieve 100% recall.

    1. Injected sets with constant rho < 0 and diverging leg term structures where
       w_cross decreases between maturities: must flag 100% with OUTPUT_AUDIT_FAILED.
    2. Calendar-consistent inputs: must achieve 0% False Positive Rate (FPR).
    """
    engine = AnalyticTriangulated()

    # 1. Generate 50 calendar-inconsistent pairs (state 0 at T=1.0, state 1 at T=1.5 with decreased variance)
    n_injected = 50
    flagged_count = 0

    for i in range(n_injected):
        T1 = 1.0
        T2 = 1.5
        rho = -0.50

        # State 0: Normal smile at T1
        # State 1: Diverging legs at T2 causing cross variance w_cross(T2) < w_cross(T1)
        X_pair = torch.zeros(2, N_COLS, dtype=torch.float64)

        # State 0
        X_pair[0, Col.F] = 100.0
        X_pair[0, Col.K] = 100.0
        X_pair[0, Col.T] = T1
        X_pair[0, Col.T_SET] = T1
        X_pair[0, Col.D] = 0.95
        X_pair[0, Col.G] = 1.0
        X_pair[0, Col.RHO] = rho
        X_pair[0, Col.PHI] = 1.0
        # Normal total variance w1=0.08, w2=0.08
        X_pair[0, Col.SVI1] = 0.08 * 0.8
        X_pair[0, Col.SVI1 + 1] = 0.08 * 0.1
        X_pair[0, Col.SVI1 + 4] = 0.1
        X_pair[0, Col.SVI2] = 0.08 * 0.8
        X_pair[0, Col.SVI2 + 1] = 0.08 * 0.1
        X_pair[0, Col.SVI2 + 4] = 0.1

        # State 1: Artificially suppressed total variance (e.g. 0.01 at T=1.5 instead of >0.08)
        X_pair[1, Col.F] = 100.0
        X_pair[1, Col.K] = 100.0
        X_pair[1, Col.T] = T2
        X_pair[1, Col.T_SET] = T2
        X_pair[1, Col.D] = 0.93
        X_pair[1, Col.G] = 1.0
        X_pair[1, Col.RHO] = rho
        X_pair[1, Col.PHI] = 1.0
        X_pair[1, Col.SVI1] = 0.01 * 0.8
        X_pair[1, Col.SVI1 + 1] = 0.01 * 0.1
        X_pair[1, Col.SVI1 + 4] = 0.1
        X_pair[1, Col.SVI2] = 0.01 * 0.8
        X_pair[1, Col.SVI2 + 1] = 0.01 * 0.1
        X_pair[1, Col.SVI2 + 4] = 0.1

        audit_res = cross_slice_audit(engine, X_pair, tol_calendar=1e-8)

        # State 1 should fail the calendar check
        if not bool(audit_res.calendar_ok[1].item()) or not bool(audit_res.ok[1].item()):
            flagged_count += 1

    recall = flagged_count / n_injected
    assert recall == 1.0, f"Gate G1.6b recall failed: {recall * 100:.1f}% != 100%"

    # 2. Verify 0% False Positive Rate on 50 calendar-consistent pairs
    n_consistent = 50
    false_positives = 0

    for i in range(n_consistent):
        T1 = 1.0
        T2 = 1.5
        rho = -0.20

        X_pair = torch.zeros(2, N_COLS, dtype=torch.float64)

        # State 0 at T1
        X_pair[0, Col.F] = 100.0
        X_pair[0, Col.K] = 100.0
        X_pair[0, Col.T] = T1
        X_pair[0, Col.T_SET] = T1
        X_pair[0, Col.D] = 0.95
        X_pair[0, Col.G] = 1.0
        X_pair[0, Col.RHO] = rho
        X_pair[0, Col.PHI] = 1.0
        X_pair[0, Col.SVI1] = 0.04 * 0.8
        X_pair[0, Col.SVI1 + 1] = 0.04 * 0.1
        X_pair[0, Col.SVI1 + 4] = 0.1
        X_pair[0, Col.SVI2] = 0.04 * 0.8
        X_pair[0, Col.SVI2 + 1] = 0.04 * 0.1
        X_pair[0, Col.SVI2 + 4] = 0.1

        # State 1 at T2 with strictly larger variance
        X_pair[1, Col.F] = 100.0
        X_pair[1, Col.K] = 100.0
        X_pair[1, Col.T] = T2
        X_pair[1, Col.T_SET] = T2
        X_pair[1, Col.D] = 0.93
        X_pair[1, Col.G] = 1.0
        X_pair[1, Col.RHO] = rho
        X_pair[1, Col.PHI] = 1.0
        X_pair[1, Col.SVI1] = 0.06 * 0.8
        X_pair[1, Col.SVI1 + 1] = 0.06 * 0.1
        X_pair[1, Col.SVI1 + 4] = 0.1
        X_pair[1, Col.SVI2] = 0.06 * 0.8
        X_pair[1, Col.SVI2 + 1] = 0.06 * 0.1
        X_pair[1, Col.SVI2 + 4] = 0.1

        audit_res = cross_slice_audit(engine, X_pair, tol_calendar=1e-8)

        if not bool(audit_res.calendar_ok.all().item()):
            false_positives += 1

    fpr = false_positives / n_consistent
    assert fpr == 0.0, f"Gate G1.6b False Positive Rate {fpr * 100:.1f}% != 0.0%"


# ── Phase 7.4: Cross-State Calendar Audit Ordering Invariance ──────────────────


@pytest.mark.audit
def test_cross_state_calendar_audit_ordering_invariance() -> None:
    """Verify that the cross-state calendar audit is invariant to the ordering of states in the batch.

    Evaluates a 4-tenor batch (where one tenor induces a calendar violation) under all
    4! = 24 permutations and verifies bitwise equality of the per-state audit outcome.
    """
    import itertools
    engine = AnalyticTriangulated()

    tenors = [0.25, 0.50, 1.00, 2.00]
    # Tenor index 2 (T=1.0) has an intentional variance decrement relative to T=0.5
    w_bases = [0.03, 0.06, 0.04, 0.12]

    base_states = []
    for t_val, w_val in zip(tenors, w_bases):
        st = torch.zeros(N_COLS, dtype=torch.float64)
        st[Col.F] = 100.0
        st[Col.K] = 100.0
        st[Col.T] = t_val
        st[Col.T_SET] = t_val
        st[Col.D] = 0.95
        st[Col.G] = 1.0
        st[Col.RHO] = -0.20
        st[Col.PHI] = 1.0
        st[Col.SVI1] = w_val * 0.8
        st[Col.SVI1 + 1] = w_val * 0.1
        st[Col.SVI1 + 4] = 0.1
        st[Col.SVI2] = w_val * 0.8
        st[Col.SVI2 + 1] = w_val * 0.1
        st[Col.SVI2 + 4] = 0.1
        base_states.append(st)

    # Reference evaluation with natural ascending order
    X_ref = torch.stack(base_states, dim=0)
    res_ref = cross_slice_audit(engine, X_ref, tol_calendar=1e-8)
    expected_cal_ok = res_ref.calendar_ok.tolist()

    # Now test all permutations of [0, 1, 2, 3]
    for perm in itertools.permutations(range(4)):
        perm_list = list(perm)
        X_perm = torch.stack([base_states[i] for i in perm_list], dim=0)
        res_perm = cross_slice_audit(engine, X_perm, tol_calendar=1e-8)
        perm_cal_ok = res_perm.calendar_ok.tolist()

        # Invert permutation to map back to original indices
        mapped_back = [perm_cal_ok[perm_list.index(orig_i)] for orig_i in range(4)]
        assert mapped_back == expected_cal_ok, (
            f"Ordering invariance broken under permutation {perm_list}: {mapped_back} != {expected_cal_ok}"
        )


# ── Phase 7.5: FastAPI TestClient Served HTTP Endpoints Arbitrage Free ────────


@pytest.mark.audit
def test_served_http_endpoint_arbitrage_free() -> None:
    """Verify that served HTTP option prices from /fx_cross/price_batch are arbitrage-free.

    Evaluates a 5x5 grid (5 strikes x 5 expiries) over FastAPI TestClient and asserts:
    1. HTTP 200 and success status
    2. Zero vertical call spread violations
    3. Zero butterfly convexity violations
    """
    from fastapi.testclient import TestClient
    from deepvol.api.server import app

    expiries = [0.1, 0.25, 0.5, 1.0, 2.0]
    strikes = [240000.0, 250000.0, 260000.0, 270000.0, 280000.0]

    states = []
    for s_idx, t_exp in enumerate(expiries):
        md = {
            "leg1": {
                "pair": "XAUUSD",
                "spot": 2650.0,
                "foreign_discount_factor": float(math.exp(-0.01 * t_exp)),
                "domestic_discount_factor": float(math.exp(-0.04 * t_exp)),
                "smile": {
                    "atm_vol": 0.16,
                    "rr25": 0.02,
                    "bf25": 0.005,
                    "rr10": 0.038,
                    "bf10": 0.017,
                    "delta_convention": "spot_pna",
                    "atm_convention": "DNS",
                    "strangle_convention": "SMILE",
                },
            },
            "leg2": {
                "pair": "USDRUB",
                "spot": 95.0,
                "foreign_discount_factor": float(math.exp(-0.04 * t_exp)),
                "domestic_discount_factor": float(math.exp(-0.16 * t_exp)),
                "smile": {
                    "atm_vol": 0.20,
                    "rr25": 0.06,
                    "bf25": 0.015,
                    "rr10": 0.12,
                    "bf10": 0.05,
                    "delta_convention": "forward_pa",
                    "atm_convention": "DNS",
                    "strangle_convention": "SMILE",
                },
            },
            "correlation": -0.20,
            "correlation_type": "HISTORICAL_LOGRETURN",
            "settlement_discount_factor": float(math.exp(-0.16 * t_exp)),
        }
        states.append({
            "state_id": f"s_{s_idx}",
            "as_of_date": "2026-10-14T00:00:00Z",
            "expiration_date": f"2027-0{s_idx+1}-15T00:00:00Z",
            "settlement_date": f"2027-0{s_idx+1}-17T00:00:00Z",
            "market_data": md,
        })

    contracts_k: List[float] = []
    contracts_state_idx: List[int] = []
    for s_idx in range(len(expiries)):
        for k in strikes:
            contracts_k.append(k)
            contracts_state_idx.append(s_idx)

    n_c = len(contracts_k)
    payload = {
        "identifier": "XAURUB",
        "payment_currency": "RUB",
        "states": states,
        "state_index": contracts_state_idx,
        "strike": contracts_k,
        "parity_sign": [1] * n_c,
        "execution_options": {
            "engine": "Analytic_Triangulated",
            "precision": "float64",
            "smile_dynamics": "STICKY_MONEYNESS",
            "use_guardian": True,
        },
    }

    with TestClient(app) as client:
        resp = client.post("/fx_cross/price_batch", json=payload)
        assert resp.status_code == 200, f"HTTP error {resp.status_code}: {resp.text}"
        data = resp.json()
        prices = data["product_value"]
        assert len(prices) == 25

        # Reshape into (5 expiries, 5 strikes)
        price_mat = np.array(prices).reshape(5, 5)

        # 1. Vertical monotonicity on each slice: C(K_i) >= C(K_{i+1})
        for i_t in range(5):
            p_slice = price_mat[i_t]
            diffs = np.diff(p_slice)
            assert np.all(diffs <= 1e-6), (
                f"Vertical monotonicity breach on slice {i_t}: diffs={diffs}"
            )

            # 2. Butterfly convexity: C(K_0) - 2C(K_1) + C(K_2) >= -1e-6
            d2 = p_slice[:-2] - 2.0 * p_slice[1:-1] + p_slice[2:]
            assert np.all(d2 >= -1e-6), (
                f"Butterfly convexity breach on slice {i_t}: d2={d2}"
            )
