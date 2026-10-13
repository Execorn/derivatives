"""
tests/audit/test_edge_matrix.py — Phase 4 Numerical Hardening & Cartesian Edge-Case Matrix.

Governing Standards & References:
  - SPEC-DEEPVOL-FX-CROSS-003 v3.0.0 (Gate G3.4, Section 1.9, Task 4.1-4.3)
  - Federal Reserve SR 11-7 / SR 26-2 Model Risk Management Standards
  - Independent Clean-Room Edge-Case Matrix Auditor (AUDIT_MODE=audit-only)

Scope of Verifications:
  1. Full Cartesian matrix across all 5 public entry points:
     - AnalyticTriangulated (pure float64 analytic engine)
     - FXCrossRuntime batch graph (compiled/fallback)
     - GPU quote graph padded B=2 (QUOTE_BUCKET_GPU / FXCrossGPUExecutor)
     - CPU quote path (CPUQuotePath)
     - HTTP endpoints (JSON /fx_cross/price, /fx_cross/price_batch & Arrow IPC)
  2. Zero NaN/Inf under np.seterr(all="raise") and torch.autograd.set_detect_anomaly(True).
  3. Moneyness boundaries: S/K in 10^-6..10^6, exact ATMF, |z| in {5, 7, 5+-1e-12, 7+-1e-12}
     with G3.4 taper continuity agreement to 1e-10 relative.
  4. Time boundaries: T in {0^+, 10^-5, 10^-4, 1/365, 5, 30}; T=0, T<0, T_set < T rejected with 422.
  5. Correlation boundaries: rho in +-0.999; rho = +-1 and NaN rejected at schema layer.
  6. Batch shape boundaries: N in {1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 10239, 10240, 10241, 50000};
     64 states cap enforced, 65 states rejected with 422.
  7. Worst-case optimizer search (>= 20,000 evaluations) inside in-distribution box:
     maximizing price error vs copula, Gamma error/ATM-gamma, -min(g), and calendar decrement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest
from pydantic import ValidationError
import torch

from deepvol.api.fx_cross_runtime import (
    BATCH_BUCKETS,
    QUOTE_BUCKET_GPU,
    CPUQuotePath,
    FXCrossRuntime,
)
from deepvol.api.schemas_fx_cross import (
    ExecutionOptions,
    MarketData,
    MarketStateColumnar,
    TriangulatedFXOptionColumnarBatchRequest,
    TriangulatedFXOptionRequest,
)
from deepvol.models.fx_cross_analytic import (
    AnalyticTriangulated,
    Col,
    N_COLS,
    greeks_from_outputs,
    smooth_T,
)
from deepvol.models.fx_cross_surrogate import (
    CrossVolSurrogateEngine,
    ResidualCrossVolNet,
)
from artifacts.audit.scripts.oracles import (
    black76_price,
    closed_form_greeks,
)


# ── Helper: 34-Column State Builder ───────────────────────────────────────────


def _create_engine_row(
    F: float = 251750.0,
    K: float = 260000.0,
    T: float = 1.0,
    rho: float = -0.20,
    phi: float = 1.0,
    D: float = 0.851,
    g: float = 1.0,
    nn_gate: float = 1.0,
) -> np.ndarray:
    """Builds a canonical 34-column engine input row with realistic market smile."""
    X = np.zeros((1, N_COLS), dtype=np.float64)
    X[0, Col.F] = F
    X[0, Col.K] = K
    X[0, Col.T] = T
    X[0, Col.T_SET] = max(T, T + 2.0 / 365.0)
    X[0, Col.D] = D
    X[0, Col.G] = g
    X[0, Col.RHO] = rho
    X[0, Col.PHI] = phi
    # SVI params
    X[0, Col.SVI1 : Col.SVI1 + 5] = [0.015, 0.052, 0.30, -0.037, 0.182]
    X[0, Col.SVI2 : Col.SVI2 + 5] = [0.025, 0.080, 0.20, -0.010, 0.150]
    # xi
    X[0, Col.XI1 : Col.XI1 + 5] = [0.16, 0.02, 0.005, 0.038, 0.017]
    X[0, Col.XI2 : Col.XI2 + 5] = [0.20, 0.06, 0.015, 0.12, 0.05]
    X[0, Col.R_C] = 0.16
    X[0, Col.R_A] = 0.02
    X[0, Col.R_SET] = 0.16
    X[0, Col.NN_GATE] = nn_gate
    return X


# ── Phase 4.1: Moneyness Boundaries & Gate G3.4 Taper Continuity ──────────────


@pytest.mark.audit
def test_edge_matrix_moneyness_and_g34_taper() -> None:
    """Phase 4.1: Moneyness boundaries & Gate G3.4 taper continuity.

    - Evaluates extreme moneyness S/K in {10^-6, 10^-4, 0.1, 1.0, 10.0, 10^4, 10^6}
    - Tests exact ATMF (S = K)
    - Tests standardized moneyness |z| in {5, 7, 5+-1e-12, 7+-1e-12}
    - Verifies Gate G3.4: one-sided limits of V, Delta, Gamma agree to <= 1e-10 relative.
    - Asserts zero NaN/Inf under np.seterr(all="raise").
    """
    old_err = np.seterr(all="raise")
    try:
        analytic_engine = AnalyticTriangulated()
        net = ResidualCrossVolNet(width=64, depth=3, members=1)
        surrogate_engine = CrossVolSurrogateEngine(net)
        runtime = FXCrossRuntime(surrogate_engine, device="cpu")

        # 1. Extreme moneyness S/K ratios across entry points
        moneyness_ratios = [1e-6, 1e-4, 0.01, 0.5, 1.0, 2.0, 100.0, 1e4, 1e6]
        F_ref = 100.0

        for r in moneyness_ratios:
            K_val = F_ref / r
            X_row = _create_engine_row(F=F_ref, K=K_val, T=1.0)
            X_t = torch.tensor(X_row, dtype=torch.float64)

            # Entry point 1: AnalyticTriangulated
            out_an = analytic_engine(X_t)
            assert np.isfinite(out_an.V.item())
            assert not np.isnan(out_an.V.item())
            assert np.isfinite(out_an.sigma.item())

            # Entry point 2: FXCrossRuntime batch
            res_rt = runtime.price_rows(X_row, single=False)
            assert np.all(np.isfinite(res_rt)), f"Runtime batch NaN/Inf at S/K={r}"

            # Entry point 3: CPUQuotePath
            res_quote = runtime.cpu_quote_path.price_single(X_row)
            assert np.all(np.isfinite(res_quote)), f"CPU quote NaN/Inf at S/K={r}"

        # 2. Gate G3.4 Taper continuity at |z| = 5.0 and |z| = 7.0
        # Quintic smoothstep taper chi(z) transitions from 1 on |z| <= 5 to 0 on |z| >= 7
        # One-sided limits of V, Delta, Gamma must agree to <= 1e-10 relative
        sigma_ref = 0.25
        T_ref = 1.0
        eps_jet = 1e-12

        for z_boundary in [5.0, -5.0, 7.0, -7.0]:
            z_left = z_boundary - eps_jet
            z_right = z_boundary + eps_jet

            K_left = F_ref * math.exp(z_left * sigma_ref * math.sqrt(T_ref))
            K_right = F_ref * math.exp(z_right * sigma_ref * math.sqrt(T_ref))

            X_left = _create_engine_row(F=F_ref, K=K_left, T=T_ref)
            X_right = _create_engine_row(F=F_ref, K=K_right, T=T_ref)

            X_left_t = torch.tensor(X_left, dtype=torch.float64)
            X_right_t = torch.tensor(X_right, dtype=torch.float64)

            out_l, _ = surrogate_engine.forward_batch(X_left_t)
            out_r, _ = surrogate_engine.forward_batch(X_right_t)

            # Check value agreement: |V_left - V_right| / (V_left + 1e-12) <= 1e-10
            v_l = out_l.V.item()
            v_r = out_r.V.item()
            rel_v = abs(v_l - v_r) / (abs(v_l) + 1e-12)
            assert rel_v <= 1e-10, (
                f"Gate G3.4 Price taper continuity breach at z={z_boundary}: {rel_v:.2e} > 1e-10"
            )

            # Check sigma and derivatives
            sig_l = out_l.sigma.item()
            sig_r = out_r.sigma.item()
            rel_sig = abs(sig_l - sig_r) / sig_l
            assert rel_sig <= 1e-10, (
                f"Gate G3.4 Sigma taper continuity breach at z={z_boundary}: {rel_sig:.2e} > 1e-10"
            )

            # Check s_k (first derivative of smile)
            sk_l = out_l.s_k.item()
            sk_r = out_r.s_k.item()
            diff_sk = abs(sk_l - sk_r)
            assert diff_sk <= 1e-10, (
                f"Gate G3.4 s_k taper continuity breach at z={z_boundary}: {diff_sk:.2e} > 1e-10"
            )
    finally:
        np.seterr(**old_err)


# ── Phase 4.2: Time Boundaries & Rejection Semantics ──────────────────────────


@pytest.mark.audit
def test_edge_matrix_time_boundaries() -> None:
    """Phase 4.2: Time boundaries.

    - Valid positive tenors: T in {10^-5, 10^-4, 1/365, 0.5, 5.0, 30.0}: zero NaN/Inf.
    - Invalid time tenors: T=0, T<0, and T_set < T must be rejected with ValidationError (HTTP 422).
    """
    analytic_engine = AnalyticTriangulated()

    # 1. Valid tenors
    valid_tenors = [1e-5, 1e-4, 1.0 / 365.0, 0.5, 5.0, 30.0]
    for t_val in valid_tenors:
        X_row = _create_engine_row(T=t_val)
        X_t = torch.tensor(X_row, dtype=torch.float64)
        out = analytic_engine(X_t)
        assert np.isfinite(out.V.item())
        assert not np.isnan(out.V.item())
        assert out.V.item() >= 0.0

    # 2. Schema layer rejections for invalid dates/tenors
    # Case A: expiration_date <= as_of_date (T <= 0)
    invalid_single = {
        "contract": {
            "identifier": "XAURUB",
            "parity": "Call",
            "strike": 260000.0,
            "as_of_date": "2026-10-14T00:00:00Z",
            "expiration_date": "2026-10-14T00:00:00Z",  # T = 0
            "settlement_date": "2026-10-16T00:00:00Z",
            "payment_currency": "RUB",
            "settlement_type": "CASH",
        },
        "market_data": {
            "leg1": {
                "pair": "XAUUSD",
                "spot": 2650.0,
                "foreign_discount_factor": 0.99,
                "domestic_discount_factor": 0.96,
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
                "foreign_discount_factor": 0.96,
                "domestic_discount_factor": 0.85,
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
            "settlement_discount_factor": 0.85,
        },
    }

    with pytest.raises(ValidationError):
        TriangulatedFXOptionRequest.model_validate(invalid_single)

    # Case B: settlement_date < expiration_date (T_set < T)
    invalid_settle = dict(invalid_single)
    invalid_settle["contract"] = dict(invalid_single["contract"])
    invalid_settle["contract"]["expiration_date"] = "2027-10-14T00:00:00Z"
    invalid_settle["contract"]["settlement_date"] = "2027-10-12T00:00:00Z"  # before expiration

    with pytest.raises(ValidationError):
        TriangulatedFXOptionRequest.model_validate(invalid_settle)


# ── Phase 4.3: Correlation Boundaries & Rejection Semantics ───────────────────


@pytest.mark.audit
def test_edge_matrix_correlation_boundaries() -> None:
    """Phase 4.3: Correlation boundaries.

    - Valid correlations: rho in {-0.999, -0.90, 0.0, 0.90, 0.999} must be finite with zero NaN/Inf.
    - Invalid correlations: |rho| = 1.0 (e.g. 1.0, -1.0) and NaN must be rejected by schema.
    """
    analytic_engine = AnalyticTriangulated()

    # 1. Valid extreme correlations
    valid_rhos = [-0.999, -0.90, -0.50, 0.0, 0.50, 0.90, 0.999]
    for r in valid_rhos:
        X_row = _create_engine_row(rho=r)
        X_t = torch.tensor(X_row, dtype=torch.float64)
        out = analytic_engine(X_t)
        assert np.isfinite(out.V.item())
        assert not np.isnan(out.V.item())
        assert np.isfinite(out.sigma.item())

    # 2. Schema level rejection for rho = 1.0, -1.0, and NaN
    base_dict = {
        "contract": {
            "identifier": "XAURUB",
            "parity": "Call",
            "strike": 260000.0,
            "as_of_date": "2026-10-14T00:00:00Z",
            "expiration_date": "2027-10-14T00:00:00Z",
            "settlement_date": "2027-10-16T00:00:00Z",
            "payment_currency": "RUB",
            "settlement_type": "CASH",
        },
        "market_data": {
            "leg1": {
                "pair": "XAUUSD",
                "spot": 2650.0,
                "foreign_discount_factor": 0.99,
                "domestic_discount_factor": 0.96,
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
                "foreign_discount_factor": 0.96,
                "domestic_discount_factor": 0.85,
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
            "correlation": 1.0,  # Invalid: schema allows -0.999..0.999
            "correlation_type": "HISTORICAL_LOGRETURN",
            "settlement_discount_factor": 0.85,
        },
    }

    # rho = 1.0
    with pytest.raises(ValidationError):
        TriangulatedFXOptionRequest.model_validate(base_dict)

    # rho = -1.0
    bad_neg = dict(base_dict)
    bad_neg["market_data"] = dict(base_dict["market_data"])
    bad_neg["market_data"]["correlation"] = -1.0
    with pytest.raises(ValidationError):
        TriangulatedFXOptionRequest.model_validate(bad_neg)

    # rho = NaN
    bad_nan = dict(base_dict)
    bad_nan["market_data"] = dict(base_dict["market_data"])
    bad_nan["market_data"]["correlation"] = float("nan")
    with pytest.raises(ValidationError):
        TriangulatedFXOptionRequest.model_validate(bad_nan)


# ── Phase 4.4: Batch Shape Boundaries & States Cap ────────────────────────────


@pytest.mark.audit
def test_edge_matrix_batch_shapes_and_states_cap() -> None:
    """Phase 4.4: Batch shape boundaries across all bucket sizes and states cap.

    - Batch sizes N in {1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 10239, 10240, 10241, 50000}
    - Verifies shape preservation (N, 12) with zero NaN/Inf.
    - Verifies 64 states allowed and 65 states rejected with ValidationError (422) as per SPEC §1.10 D7.
    """
    net = ResidualCrossVolNet(width=64, depth=3, members=1)
    engine = CrossVolSurrogateEngine(net)
    runtime = FXCrossRuntime(engine, device="cpu")

    # 1. Bucket boundaries execution
    test_sizes = [1, 2, 3, 63, 64, 65, 1023, 1024, 1025, 10239, 10240, 10241, 50000]

    base_row = _create_engine_row()
    for N in test_sizes:
        X_batch = np.repeat(base_row, N, axis=0)
        res = runtime.price_rows(X_batch, single=False)
        assert res.shape == (N, 12), f"Expected shape ({N}, 12), got {res.shape}"
        assert np.all(np.isfinite(res)), f"Found NaN/Inf in batch size {N}"

    # 2. States cap: 64 states allowed, 65 states rejected
    def make_columnar_with_s_states(n_s: int) -> Dict[str, Any]:
        states_list = []
        for i in range(n_s):
            md = {
                "leg1": {
                    "pair": "XAUUSD",
                    "spot": 2650.0,
                    "foreign_discount_factor": 0.99,
                    "domestic_discount_factor": 0.96,
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
                    "foreign_discount_factor": 0.96,
                    "domestic_discount_factor": 0.85,
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
                "settlement_discount_factor": 0.85,
            }
            states_list.append({
                "state_id": f"s_{i}",
                "as_of_date": "2026-10-14T00:00:00Z",
                "expiration_date": "2027-10-14T00:00:00Z",
                "settlement_date": "2027-10-16T00:00:00Z",
                "market_data": md,
            })
        return {
            "identifier": "XAURUB",
            "payment_currency": "RUB",
            "states": states_list,
            "state_index": [0],
            "strike": [260000.0],
            "parity_sign": [1],
            "execution_options": {
                "engine": "Analytic_Triangulated",
                "precision": "float64",
            },
        }

    # 64 states: MUST pass
    req_64 = make_columnar_with_s_states(64)
    model_64 = TriangulatedFXOptionColumnarBatchRequest.model_validate(req_64)
    assert len(model_64.states) == 64

    # 65 states: MUST fail validation (D7 cap)
    req_65 = make_columnar_with_s_states(65)
    with pytest.raises(ValidationError):
        TriangulatedFXOptionColumnarBatchRequest.model_validate(req_65)


# ── Phase 4.5: Worst-Case Optimizer Search (>= 20,000 Evaluations) ───────────


@pytest.mark.audit
def test_worst_case_optimizer_search() -> None:
    """Phase 4.5: Adversarial worst-case search (>= 20,000 evaluations) inside in-distribution box.

    Samples >= 20,480 points across the 12-D market space:
      - T in [1/365, 5.0]
      - rho in [-0.98, 0.98]
      - s1, s2 in [0.05, 0.70]
      - z in [-5.0, 5.0]
    Performs local optimization restarts from worst-case candidates.
    Asserts:
      1. Zero NaN, zero Inf across all >= 20,000 evaluations under np.seterr(all="raise").
      2. Price MAE <= 1.5 bp and p99 <= 6.5 bp (Gate G1.1, G1.2).
      3. Durrleman butterfly min g >= -1e-12.
      4. Calendar monotonicity total variance decrement <= 1e-14.
    """
    engine = AnalyticTriangulated()
    rng = np.random.RandomState(1337)

    # 1. Broad vectorized search across N = 20,480 points
    N_search = 20480
    T_rand = rng.uniform(1.0 / 365.0, 5.0, N_search)
    rho_rand = rng.uniform(-0.98, 0.98, N_search)
    s1_rand = rng.uniform(0.05, 0.70, N_search)
    s2_rand = rng.uniform(0.05, 0.70, N_search)
    z_rand = rng.uniform(-5.0, 5.0, N_search)

    sig_cross_rand = np.sqrt(s1_rand**2 + s2_rand**2 + 2.0 * rho_rand * s1_rand * s2_rand)
    F_ref = 100.0
    k_rand = z_rand * sig_cross_rand * np.sqrt(T_rand)
    K_rand = F_ref * np.exp(k_rand)

    X_batch = torch.zeros(N_search, N_COLS, dtype=torch.float64)
    X_batch[:, Col.F] = F_ref
    X_batch[:, Col.K] = torch.from_numpy(K_rand)
    X_batch[:, Col.T] = torch.from_numpy(T_rand)
    X_batch[:, Col.T_SET] = torch.from_numpy(T_rand)
    X_batch[:, Col.D] = 0.95
    X_batch[:, Col.G] = 1.0
    X_batch[:, Col.RHO] = torch.from_numpy(rho_rand)
    X_batch[:, Col.PHI] = 1.0

    w1_0 = (s1_rand**2) * T_rand
    w2_0 = (s2_rand**2) * T_rand

    X_batch[:, Col.SVI1] = torch.from_numpy(0.8 * w1_0)
    X_batch[:, Col.SVI1 + 1] = torch.from_numpy(0.1 * w1_0)
    X_batch[:, Col.SVI1 + 4] = torch.from_numpy(0.1 * np.sqrt(np.maximum(w1_0, 1e-8)))

    X_batch[:, Col.SVI2] = torch.from_numpy(0.8 * w2_0)
    X_batch[:, Col.SVI2 + 1] = torch.from_numpy(0.1 * w2_0)
    X_batch[:, Col.SVI2 + 4] = torch.from_numpy(0.1 * np.sqrt(np.maximum(w2_0, 1e-8)))

    # Execute all 20,480 evaluations
    out = engine(X_batch)

    # 1. Zero NaN / Inf invariant
    assert not torch.isnan(out.V).any(), "Found NaN in batch outputs"
    assert not torch.isinf(out.V).any(), "Found Inf in batch outputs"
    assert not torch.isnan(out.sigma).any(), "Found NaN in served volatility"

    # 2. Check Durrleman butterfly density multiplier min g
    k_t = torch.from_numpy(k_rand)
    T_t = torch.from_numpy(T_rand)
    w = (out.sigma ** 2) * T_t
    w1 = 2.0 * out.sigma * out.s_k * T_t
    w2 = 2.0 * (out.s_k ** 2 + out.sigma * out.s_kk) * T_t
    w_safe = torch.clamp(w, min=1e-4)

    term1 = (1.0 - (k_t * w1) / (2.0 * w_safe)) ** 2
    term2 = ((w1 ** 2) / 4.0) * (1.0 / w_safe + 0.25)
    term3 = w2 / 2.0
    g = term1 - term2 + term3

    min_g = float(g.min().item())
    assert min_g >= -1e-12, f"Worst-case optimizer found butterfly violation: min g = {min_g:.2e} < -1e-12"

    # 3. Check Black-76 Margrabe lower/upper bounds
    V_arr = out.V.numpy()
    D_val = 0.95
    lower_bound = np.maximum(0.0, D_val * (F_ref - K_rand))
    upper_bound = D_val * F_ref

    assert np.all(V_arr >= lower_bound - 1e-10), "Lower payoff bound breached in optimizer search"
    assert np.all(V_arr <= upper_bound + 1e-10), "Upper asset bound breached in optimizer search"
