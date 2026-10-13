"""test_mrm_audit.py — Adversarial MRM, Governance, and Compliance Audit Suite (Phase 12).

Governing Standards:
  - Federal Reserve SR 11-7: Model Risk Management (MRM)
  - Federal Reserve SR 26-2: Artificial Intelligence & Machine Learning Model Governance
  - SPEC-DEEPVOL-FX-CROSS-003: Phase 12 (Tasks 3.3, 3.4, Gate G3.2, Gate G3.3)

Scope:
  1. FXCrossGuardian.route:
     - 12-D Ledoit-Wolf Mahalanobis distance & bounding box checks.
     - Exhaustive fallback reason codes (EXTREME_MONEYNESS, OOD_MARKET_REGIME,
       EPISTEMIC_UNCERTAINTY, SMILE_ARBITRAGE, OUTPUT_AUDIT_FAILED, ENGINE_REQUESTED).
  2. ModelReserveTracker:
     - Deterministic 64-strike grid (56 core + 8 taper).
     - MR_s formula: max_K |V_cop - V_served| / (D * F) * 1e4 bp.
     - Thread-safe TTL cache, monotonic hot path, non-blocking asynchronous refresh.
  3. PSI drift monitor:
     - Rolling Population Stability Index (PSI) tracking over 12 features.
     - 10 bins, 0.25 critical alert threshold.
  4. Model card & gate report validation:
     - Schema compliance, cryptographic data manifest hash match.
     - Independent consistency check of G1-G4 metrics.
  5. Defect reproductions (test_F012, test_F013, test_F014):
     - F012: Omission of SR 11-7 structured compliance audit logging on fallback/OOD.
     - F013: Detached drift monitor: FXCrossGuardian vs mrm.compliance.DriftMonitor.
     - F014: Discrepancy between server.py model_version and model_card.json.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch import Tensor

from deepvol.market.svi_slice import RawSVI
from deepvol.models.fx_cross_analytic import Col, N_COLS
from deepvol.models.fx_cross_copula import (
    CopulaConfig,
    CrossState,
    GaussianCopulaCrossEngine,
)
from deepvol.models.fx_cross_safeguards import AuditResult
from deepvol.mrm.fx_cross_guardian import (
    FallbackReasonEnum,
    FXCrossGuardian,
    MarketStateDriftMonitor,
    MarketStateOODDetector,
    ModelReserveEntry,
    ModelReserveTracker,
    PricingEngineEnum,
    RoutingDecision,
)


@pytest.mark.audit
def test_mrm_guardian_route_ood_mahalanobis() -> None:
    """Verifies 12-D Ledoit-Wolf Mahalanobis distance and bounding box checks."""
    model_card_path = Path("artifacts/models/fx_cross/model_card.json")
    assert model_card_path.exists()

    detector = MarketStateOODDetector.from_checkpoint(model_card_path)
    assert detector.mean.shape == (12,)
    assert detector.cov_inv.shape == (12, 12)
    assert detector.q995 > 0.0

    # 1. Test In-Distribution state (exactly at training mean)
    mean_state = detector.mean.clone().unsqueeze(0)
    d_m, box_viol, is_ood = detector.score(mean_state)
    assert float(d_m[0].item()) < 1e-4
    assert not bool(box_viol[0].item())
    assert not bool(is_ood[0].item())

    # 2. Test Extreme Mahalanobis state (outside q99.5 ellipsoid, inside box)
    # Scale along principal axis of covariance
    eigenvalues, eigenvectors = torch.linalg.eigh(detector.cov_inv)
    min_eig_vec = eigenvectors[:, 0]
    scaled_offset = min_eig_vec * (detector.q995 * 2.0)
    ood_mahal_state = (detector.mean + scaled_offset).unsqueeze(0)

    d_m_ood, _, is_ood_mahal = detector.score(ood_mahal_state)
    assert float(d_m_ood[0].item()) > detector.q995
    assert bool(is_ood_mahal[0].item())

    # 3. Test Bounding Box violation (extreme single feature)
    box_viol_state = detector.mean.clone().unsqueeze(0)
    box_viol_state[0, 0] = detector.box_hi[0] * 5.0  # sqrt_T blown up
    _, box_viol_flag, is_ood_box = detector.score(box_viol_state)
    assert bool(box_viol_flag[0].item())
    assert bool(is_ood_box[0].item())


@pytest.mark.audit
def test_mrm_guardian_routing_reasons_exhaustive() -> None:
    """Verifies all FallbackReasonEnum routing states and engine transitions."""
    model_card_path = Path("artifacts/models/fx_cross/model_card.json")
    detector = MarketStateOODDetector.from_checkpoint(model_card_path)
    guardian = FXCrossGuardian(ood_detector=detector, enable_slice_audit=False)

    # Construct test 12-D states
    in_dist = detector.mean.clone().unsqueeze(0)
    ood_state = (detector.mean + 100.0).unsqueeze(0)
    batch_states = torch.cat([in_dist, ood_state], dim=0)

    # 1. Differential_ML on mixed batch
    decision = guardian.route(batch_states, engine=PricingEngineEnum.DIFFERENTIAL_ML)
    assert decision.nn_gate[0].item() == 1.0
    assert not decision.fallback_activated[0].item()
    assert decision.fallback_reason[0] is None
    assert decision.engine_used[0] == PricingEngineEnum.DIFFERENTIAL_ML

    assert decision.nn_gate[1].item() == 0.0
    assert decision.fallback_activated[1].item()
    assert decision.fallback_reason[1] == FallbackReasonEnum.OOD_MARKET_REGIME.value
    assert decision.engine_used[1] == PricingEngineEnum.ANALYTIC_TRIANGULATED

    # 2. Engine requested Analytic_Triangulated
    dec_analytic = guardian.route(in_dist, engine=PricingEngineEnum.ANALYTIC_TRIANGULATED)
    assert dec_analytic.nn_gate[0].item() == 0.0
    assert dec_analytic.fallback_activated[0].item()
    assert dec_analytic.fallback_reason[0] == FallbackReasonEnum.ENGINE_REQUESTED.value
    assert dec_analytic.engine_used[0] == PricingEngineEnum.ANALYTIC_TRIANGULATED

    # 3. Post-check epistemic uncertainty failover
    eps_std_bp = torch.tensor([5.0, 0.5], dtype=torch.float64)
    reserve_bp = torch.tensor([10.0, 10.0], dtype=torch.float64)
    dec_post = guardian.apply_post_check(decision, eps_std_bp, reserve_bp)
    assert dec_post.nn_gate[0].item() == 0.0
    assert dec_post.fallback_activated[0].item()
    assert dec_post.fallback_reason[0] == FallbackReasonEnum.EPISTEMIC_UNCERTAINTY.value
    assert dec_post.engine_used[0] == PricingEngineEnum.ANALYTIC_TRIANGULATED

    # 4. Output audit failed failover
    mock_audit = AuditResult(
        min_g=torch.tensor([-0.01, -0.01], dtype=torch.float64),
        call_spread_ok=torch.tensor([False, False]),
        slope_bound=torch.tensor([1.5, 1.5], dtype=torch.float64),
        ok=torch.tensor([False, False]),
        calendar_ok=torch.tensor([True, True]),
    )
    dec_audit = guardian.apply_audit_check(decision, mock_audit)
    # State 0 was valid, now OUTPUT_AUDIT_FAILED
    assert dec_audit.nn_gate[0].item() == 0.0
    assert dec_audit.fallback_activated[0].item()
    assert dec_audit.fallback_reason[0] == FallbackReasonEnum.OUTPUT_AUDIT_FAILED.value

    # State 1 was upstream OOD_MARKET_REGIME -> reason preserved!
    assert dec_audit.fallback_reason[1] == FallbackReasonEnum.OOD_MARKET_REGIME.value


@pytest.mark.audit
def test_mrm_reserve_tracker_calculation_and_caching() -> None:
    """Verifies ModelReserveTracker 64-strike grid, MR_s formula, and thread-safe caching."""
    cfg = CopulaConfig(n_nodes=101, strike_chunk=64, device="cpu")
    copula = GaussianCopulaCrossEngine(cfg=cfg)
    tracker = ModelReserveTracker(copula=copula, max_age=timedelta(seconds=2))

    # Construct mock CrossState
    svi_params = torch.tensor([[0.04, 0.1, -0.2, 0.0, 0.1]], dtype=torch.float64)
    state = CrossState(
        F1=torch.tensor([1.10], dtype=torch.float64),
        F2=torch.tensor([1.15], dtype=torch.float64),
        T=torch.tensor([0.5], dtype=torch.float64),
        svi1=RawSVI(svi_params),
        svi2=RawSVI(svi_params),
        rho=torch.tensor([0.3], dtype=torch.float64),
    )

    # 1. Verify 64 strikes generated
    strikes = tracker.generate_strikes(state)
    assert strikes.squeeze().shape == (64,)
    # Strikes must be strictly increasing
    assert torch.all(torch.diff(strikes.squeeze()) > 0)

    # 2. Compute reserve with mock served engine (producing constant 1 bp error)
    F_cross = float(state.F[0].item())
    def served_fn(k: Tensor) -> Tensor:
        cop_undisc = copula.price(state, k, is_call=True)
        # Introduce exactly 0.0001 * F error
        return cop_undisc + (1e-4 * F_cross)

    mr_bp = tracker.compute_reserve(state, served_fn)
    # Error is 1e-4 * F, normalized by (D * F) * 1e4 -> exactly 1.0 bp
    assert abs(mr_bp - 1.0) < 1e-3

    # 3. Test caching and staleness
    key = "EUR_USD_0.5y"
    val, stale = tracker.get(key)
    assert val is None
    assert stale is True

    # Run async refresh
    async def run_refresh():
        await tracker.refresh(key, state, served_fn)

    asyncio.run(run_refresh())

    cached_val, stale_after = tracker.get(key)
    assert cached_val is not None
    assert abs(cached_val - 1.0) < 1e-3
    assert stale_after is False

    # Simulate expiry
    time.sleep(2.1)
    expired_val, stale_expired = tracker.get(key)
    assert expired_val is not None
    assert stale_expired is True


@pytest.mark.audit
def test_mrm_psi_drift_monitor_bins_and_threshold() -> None:
    """Verifies MarketStateDriftMonitor rolling window, 10-bin PSI, and 0.25 trigger."""
    monitor = MarketStateDriftMonitor(window_size=200, psi_alert_threshold=0.25)

    # Generate synthetic baseline: 1000 samples of 12 standard normal features
    rng = np.random.RandomState(42)
    baseline = rng.randn(1000, 12)
    monitor.set_baseline(baseline)

    # 1. Record in-distribution samples drawn from exact same distribution
    in_dist_samples = rng.randn(200, 12)
    monitor.record(in_dist_samples)

    psi_map = monitor.compute_psi(num_bins=10)
    assert len(psi_map) == 12
    # All in-distribution PSIs should be comfortably below 0.25 alert threshold
    for feat, psi_val in psi_map.items():
        assert psi_val < 0.25, f"Feature {feat} exceeded 0.25 on in-distribution data: {psi_val}"
    assert not monitor.is_drift_detected()

    # 2. Record strongly drifted samples (shifted mean by +3.5)
    drifted_samples = rng.randn(200, 12) + 3.5
    monitor.record(drifted_samples)

    psi_drift_map = monitor.compute_psi(num_bins=10)
    assert monitor.is_drift_detected()
    # At least one feature has PSI >= 0.25
    max_psi = max(psi_drift_map.values())
    assert max_psi >= 0.25


@pytest.mark.audit
def test_mrm_model_card_and_gate_json_schema_validity() -> None:
    """Verifies model_card.json and fx_cross_gates.json schema, hashes, and gate thresholds."""
    model_card_p = Path("artifacts/models/fx_cross/model_card.json")
    gates_p = Path("artifacts/reports/fx_cross_gates.json")
    manifest_p = Path("artifacts/data/fx_cross_shards/manifest.json")

    assert model_card_p.exists()
    assert gates_p.exists()
    assert manifest_p.exists()

    with open(model_card_p, "r", encoding="utf-8") as f:
        card = json.load(f)

    with open(gates_p, "r", encoding="utf-8") as f:
        gates = json.load(f)

    with open(manifest_p, "rb") as f:
        manifest_sha256 = hashlib.sha256(f.read()).hexdigest()

    # 1. Cryptographic manifest hash match
    assert card["data_manifest_hash"] == manifest_sha256, (
        f"model_card data_manifest_hash {card['data_manifest_hash']} does not match "
        f"computed manifest.json sha256 {manifest_sha256}"
    )

    # 2. Gate report integrity
    g1_criteria = gates["gates"]["G1"]["criteria"]
    expected_gates = ["G1.1", "G1.2", "G1.3", "G1.3b", "G1.4", "G1.5a", "G1.5b"]
    for g_id in expected_gates:
        assert g_id in g1_criteria, f"Missing gate {g_id} in fx_cross_gates.json"
        gate_data = g1_criteria[g_id]
        assert gate_data["passed"] is True
        assert gate_data["measured"] <= gate_data["threshold"]

    # 3. Model card feature statistics sanity
    feat_stats = card["feature_statistics"]
    assert len(feat_stats["mean"]) == 13
    assert len(feat_stats["std"]) == 13
    assert all(s > 0 for s in feat_stats["std"])


@pytest.mark.audit
def test_F012_sr11_7_audit_logging_omitted() -> None:
    """DEFECT REPRODUCTION: Omission of SR 11-7 structured compliance audit logging on fallback/OOD.

    Spec Reference:
      - Phase 12 item 5: "SR 11-7 audit logging: Fallback and OOD decisions logged with
        request ID, reason, model version. Zero PII or secrets."

    Root Cause:
      In `src/deepvol/api/server.py` (_fx_cross_price_impl, _fx_cross_price_batch_impl) and
      `src/deepvol/mrm/fx_cross_guardian.py` (FXCrossGuardian.route), when a fallback is triggered
      (e.g., OOD_MARKET_REGIME, EXTREME_MONEYNESS, OUTPUT_AUDIT_FAILED), NO structured compliance log
      is emitted. SR 11-7 model governance mandates immutable audit trail logging containing
      request_id, fallback_reason, and model_version for model risk examination.
    """
    src_server = Path("src/deepvol/api/server.py").read_text(encoding="utf-8")
    src_guardian = Path("src/deepvol/mrm/fx_cross_guardian.py").read_text(encoding="utf-8")

    # Check whether server or guardian emits SR 11-7 audit logs on fallback
    has_audit_log_server = "SR11_7" in src_server or "fallback_reason" in src_server and "logger.info" in src_server
    has_audit_log_guardian = "SR11_7" in src_guardian or "logger.warning" in src_guardian and "request_id" in src_guardian

    assert has_audit_log_server or has_audit_log_guardian, (
        "F012 VIOLATION: server.py and FXCrossGuardian.route omit SR 11-7 structured compliance audit logging "
        "when fallback or OOD routing is activated (no logging of request_id, fallback_reason, or model_version)"
    )


@pytest.mark.audit
def test_F013_drift_monitor_detached_from_mrm_compliance() -> None:
    """DEFECT REPRODUCTION: FXCrossGuardian MarketStateDriftMonitor is detached from mrm.compliance.

    Spec Reference:
      - Phase 12 item 3: "PSI drift monitoring: Wired to mrm/compliance.DriftMonitor.
        Check bins, thresholds (>0.25 trigger), alert paths."

    Root Cause:
      `src/deepvol/mrm/compliance.py` implements the central SR 26-2 compliance monitoring
      with `DriftMonitor` and `check_compliance()`, but it is hardcoded to 6 Heston parameters
      (`RH_BOUNDS` / `PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]`).
      Meanwhile, `MarketStateDriftMonitor` in `src/deepvol/mrm/fx_cross_guardian.py` operates as an
      isolated local class and never feeds or connects to `mrm.compliance.DriftMonitor` or `_global_monitor`.
    """
    from deepvol.mrm import compliance
    from deepvol.mrm.fx_cross_guardian import FEATURE_NAMES_12D, FXCrossGuardian

    guardian = FXCrossGuardian()

    # The spec requires wiring to mrm.compliance.DriftMonitor
    is_wired_to_compliance = isinstance(guardian.drift_monitor, compliance.DriftMonitor)

    # Furthermore, compliance.DriftMonitor has no awareness of 12-D FX cross features
    has_12d_in_compliance = hasattr(compliance, "FEATURE_NAMES_12D") or any(
        f in compliance.PARAM_NAMES for f in FEATURE_NAMES_12D
    )

    assert is_wired_to_compliance and has_12d_in_compliance, (
        "F013 VIOLATION: MarketStateDriftMonitor in fx_cross_guardian is detached from "
        "deepvol.mrm.compliance.DriftMonitor, and compliance.py remains hardcoded to 6 Heston parameters"
    )


@pytest.mark.audit
def test_F014_model_version_inconsistency_server_vs_model_card() -> None:
    """DEFECT REPRODUCTION: Inconsistency between server.py model_version and model_card.json.

    Spec Reference:
      - Phase 12 item 4: "model_card.json & artifacts/reports/fx_cross_gates.json:
        data manifest hash matches shards, model_version consistent."

    Root Cause:
      In `src/deepvol/api/server.py:2147`:
      `_FX_CROSS_STATE = { ... "model_version": "deepvol-fx-cross-v3.0-b24977e7" }`
      However, the production release artifacts (`artifacts/models/fx_cross/model_card.json`
      and `residual_vol_v3_m0.pt`) specify:
      `"model_version": "1326b33ce012+afc6d3f"`
      This discrepancy causes model telemetry in API responses to report an untracked/stale model version.
    """
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    from deepvol.api import server

    model_card_p = Path("artifacts/models/fx_cross/model_card.json")
    with open(model_card_p, "r", encoding="utf-8") as f:
        card = json.load(f)

    server_version = server._FX_CROSS_STATE["model_version"]
    card_version = card["model_version"]

    assert server_version == card_version, (
        f"F014 VIOLATION: server.py hardcodes model_version '{server_version}' which contradicts "
        f"official model_card.json version '{card_version}'"
    )
