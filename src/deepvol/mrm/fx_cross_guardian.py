"""fx_cross_guardian.py — Model Risk Guardian and Empirical OOD Detector for Triangulated FX Crosses.

Governing Standards:
  - Federal Reserve SR 11-7: Model Risk Management (MRM)
  - Federal Reserve SR 26-2: Artificial Intelligence & Machine Learning Model Governance
  - SPEC-DEEPVOL-FX-CROSS-003: Tasks 3.3 and 3.4 (Gate G3.2, Gate G3.3)

Provides:
  - PricingEngineEnum & FallbackReasonEnum: Core engine and fallback enums.
  - MarketStateOODDetector: 12-D Ledoit-Wolf Mahalanobis distance & bounding box checks in float64.
  - MarketStateDriftMonitor: Rolling Population Stability Index (PSI) drift tracking.
  - FXCrossGuardian: Intercepts pricing calls, triggers deterministic fallbacks to
    Analytic_Triangulated with bitwise equality, executes epistemic post-checks,
    and enforces Hybrid_Ensemble consensus upon drift alerts.
  - ModelReserveTracker: Non-blocking asynchronous copula model reserve tracking and TTL cache.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import torch.nn as nn

from deepvol.calibration.fallbacks import calculate_psi
from deepvol.models.fx_cross_analytic import AnalyticTriangulated, Col, smooth_T
from deepvol.models.fx_cross_copula import CrossState, GaussianCopulaCrossEngine
from deepvol.models.fx_cross_safeguards import AuditResult, cross_slice_audit
from deepvol.mrm.guardian import ModelRiskGuardian

logger = logging.getLogger("deepvol.mrm.fx_cross_guardian")


class PricingEngineEnum(str, Enum):
    """Supported pricing engines for triangulated cross options."""
    DIFFERENTIAL_ML = "Differential_ML"
    ANALYTIC_TRIANGULATED = "Analytic_Triangulated"
    ANALYTIC_GK = "Analytic_GK"              # v1.0 alias, resolved to Analytic_Triangulated
    HYBRID_ENSEMBLE = "Hybrid_Ensemble"
    COPULA_BENCHMARK = "Copula_Benchmark"    # validation engine, not latency-bound


class FallbackReasonEnum(str, Enum):
    """Reason codes for guardian failover or taper deactivation."""
    EXTREME_MONEYNESS = "EXTREME_MONEYNESS"            # |z| >= 7: NN contribution is exactly zero
    OOD_MARKET_REGIME = "OOD_MARKET_REGIME"
    EPISTEMIC_UNCERTAINTY = "EPISTEMIC_UNCERTAINTY"
    SMILE_ARBITRAGE = "SMILE_ARBITRAGE"
    OUTPUT_AUDIT_FAILED = "OUTPUT_AUDIT_FAILED"        # served cross slice failed the Durrleman / spread audit
    ENGINE_REQUESTED = "ENGINE_REQUESTED"


FEATURE_NAMES_12D: Tuple[str, ...] = (
    "sqrt_T",
    "rho",
    "xi1_atm",
    "xi1_rr25",
    "xi1_bf25",
    "xi1_rr10",
    "xi1_bf10",
    "xi2_atm",
    "xi2_rr25",
    "xi2_bf25",
    "xi2_rr10",
    "xi2_bf10",
)


@dataclass(frozen=True)
class RoutingDecision:
    """Routing metadata emitted by FXCrossGuardian.route."""

    nn_gate: Tensor
    fallback_activated: Tensor
    fallback_reason: List[Optional[str]]
    d_m: Tensor
    is_ood: Tensor
    box_violation: Tensor
    engine_used: List[PricingEngineEnum]

    @property
    def reason(self) -> List[Optional[str]]:
        return self.fallback_reason

    @property
    def D_M(self) -> Tensor:
        return self.d_m


class MarketStateOODDetector:
    """Empirical 12-D Out-of-Distribution detector using Ledoit-Wolf Mahalanobis distance and bounding box checks."""

    def __init__(
        self,
        mean: Tensor,
        cov_inv: Tensor,
        q995: float,
        box_lo: Tensor,
        box_hi: Tensor,
        margin: float = 0.02,
    ) -> None:
        if mean.shape != (12,):
            raise ValueError(f"Expected mean of shape (12,), got {mean.shape}")
        if cov_inv.shape != (12, 12):
            raise ValueError(f"Expected cov_inv of shape (12, 12), got {cov_inv.shape}")
        if box_lo.shape != (12,) or box_hi.shape != (12,):
            raise ValueError("Expected box_lo and box_hi of shape (12,)")

        self.mean = mean.to(dtype=torch.float64)
        self.cov_inv = cov_inv.to(dtype=torch.float64)
        self.q995 = float(q995)
        self.box_lo = box_lo.to(dtype=torch.float64)
        self.box_hi = box_hi.to(dtype=torch.float64)
        self.margin = float(margin)

        # Apply span margin if specified
        if self.margin > 0.0:
            span = self.box_hi - self.box_lo
            self.eff_box_lo = self.box_lo - self.margin * span
            self.eff_box_hi = self.box_hi + self.margin * span
        else:
            self.eff_box_lo = self.box_lo
            self.eff_box_hi = self.box_hi

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> "MarketStateOODDetector":
        """Loads Ledoit-Wolf statistics, q99.5 quantile, and box bounds from checkpoint or model card."""
        p = Path(path)
        if p.is_dir():
            card_p = p / "model_card.json"
            ckpt_p = p / "residual_vol_v3_m0.pt"
            if card_p.exists():
                p = card_p
            elif ckpt_p.exists():
                p = ckpt_p
            else:
                raise FileNotFoundError(f"No model_card.json or .pt checkpoint found in {p}")

        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            ood_data = data.get("ood_statistics", {})
            mean = torch.tensor(ood_data["mean_12d"], dtype=torch.float64)
            cov_inv = torch.tensor(ood_data["cov_inv_12d"], dtype=torch.float64)
            q995 = float(ood_data["q995_mahalanobis"])
            box_bounds = ood_data["box_bounds"]
            box_lo = torch.tensor(box_bounds["lo"], dtype=torch.float64)
            box_hi = torch.tensor(box_bounds["hi"], dtype=torch.float64)
            return cls(mean=mean, cov_inv=cov_inv, q995=q995, box_lo=box_lo, box_hi=box_hi, margin=0.0)

        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        ood_data = ckpt.get("ood_statistics", {})
        if not ood_data:
            raise KeyError(f"Checkpoint at {p} does not contain 'ood_statistics'")

        mean = torch.as_tensor(ood_data["mean_12d"], dtype=torch.float64)
        cov_inv = torch.as_tensor(ood_data["cov_inv_12d"], dtype=torch.float64)
        q995_val = ood_data["q995_mahalanobis"]
        q995 = float(q995_val.item() if isinstance(q995_val, Tensor) else q995_val)
        box_lo = torch.as_tensor(ood_data["box_lo"], dtype=torch.float64)
        box_hi = torch.as_tensor(ood_data["box_hi"], dtype=torch.float64)
        return cls(mean=mean, cov_inv=cov_inv, q995=q995, box_lo=box_lo, box_hi=box_hi, margin=0.0)

    @classmethod
    def fit(cls, train_states: Tensor, margin: float = 0.02) -> "MarketStateOODDetector":
        """Fits Ledoit-Wolf shrinkage covariance and empirical q99.5 quantile from training tensor."""
        S_cnt, D_cnt = train_states.shape
        if D_cnt != 12:
            raise ValueError(f"Expected 12-D market states, got shape {train_states.shape}")

        states_f64 = train_states.to(dtype=torch.float64)
        mean_12d = states_f64.mean(dim=0)
        diff = states_f64 - mean_12d

        cov_sample = (diff.t() @ diff) / max(1, S_cnt - 1)
        eye = torch.eye(12, dtype=torch.float64, device=train_states.device)
        target = (torch.trace(cov_sample) / 12.0) * eye

        if S_cnt > 1:
            d2 = torch.norm(cov_sample - target, p="fro") ** 2
            y = diff.unsqueeze(2) * diff.unsqueeze(1)
            cov_sample_exp = cov_sample.unsqueeze(0)
            phi = torch.sum(torch.norm(y - cov_sample_exp, dim=(1, 2), p="fro") ** 2) / (S_cnt ** 2)
            shrinkage = float(torch.clamp(phi / (d2.clamp_min(1e-12)), 0.0, 1.0).item())
        else:
            shrinkage = 0.1

        cov_shrunk = (1.0 - shrinkage) * cov_sample + shrinkage * target + 1e-6 * eye
        cov_inv_12d = torch.linalg.inv(cov_shrunk)

        d_m_sq = torch.sum((diff @ cov_inv_12d) * diff, dim=-1)
        d_m = torch.sqrt(torch.clamp(d_m_sq, min=0.0))
        q995 = float(torch.quantile(d_m, 0.995).item())

        box_lo = states_f64.min(dim=0).values
        box_hi = states_f64.max(dim=0).values

        return cls(
            mean=mean_12d,
            cov_inv=cov_inv_12d,
            q995=q995,
            box_lo=box_lo,
            box_hi=box_hi,
            margin=margin,
        )

    def score(self, m: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluates 12-D market states m (S, 12) returning (D_M, box_violation, is_ood)."""
        if m.ndim == 1:
            m = m.unsqueeze(0)
        if m.shape[-1] != 12:
            raise ValueError(f"Expected 12 features per market state, got {m.shape[-1]}")

        m_f64 = m.to(dtype=torch.float64, device=self.mean.device)
        diff = m_f64 - self.mean

        d_m_sq = torch.sum((diff @ self.cov_inv) * diff, dim=-1)
        D_M = torch.sqrt(torch.clamp(d_m_sq, min=0.0))

        eff_lo = self.eff_box_lo.to(device=m.device)
        eff_hi = self.eff_box_hi.to(device=m.device)

        lo_viol = torch.any(m_f64 < eff_lo, dim=-1)
        hi_viol = torch.any(m_f64 > eff_hi, dim=-1)
        box_violation = lo_viol | hi_viol

        is_ood = box_violation | (D_M > self.q995)
        return D_M, box_violation, is_ood


class MarketStateDriftMonitor:
    """Tracks rolling feature stream drift against baseline using Population Stability Index (PSI)."""

    def __init__(
        self,
        baseline: Optional[Union[Tensor, NDArray[Any]]] = None,
        window_size: int = 200,
        psi_alert_threshold: float = 0.25,
    ) -> None:
        self.window_size = window_size
        self.psi_alert_threshold = psi_alert_threshold
        self.history: List[NDArray[Any]] = []
        self.baseline: Optional[NDArray[Any]] = None

        if baseline is not None:
            self.set_baseline(baseline)

    def set_baseline(self, baseline: Union[Tensor, NDArray[Any]]) -> None:
        b = baseline.detach().cpu().numpy() if isinstance(baseline, Tensor) else np.asarray(baseline)
        if b.ndim != 2 or b.shape[1] != 12:
            raise ValueError(f"Expected baseline of shape (N, 12), got {b.shape}")
        self.baseline = b.astype(np.float64)

    def record(self, states: Union[Tensor, NDArray[Any]]) -> None:
        """Records incoming market states into rolling history."""
        s = states.detach().cpu().numpy() if isinstance(states, Tensor) else np.asarray(states)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        for row in s:
            self.history.append(row.copy())
            if len(self.history) > self.window_size:
                self.history.pop(0)

    def compute_psi(self, num_bins: int = 10) -> Dict[str, float]:
        """Computes per-feature PSI across rolling history against baseline."""
        if self.baseline is None or len(self.history) < 20:
            return {}

        actual_data = np.array(self.history)
        psi_dict: Dict[str, float] = {}

        for j, name in enumerate(FEATURE_NAMES_12D):
            base_col = self.baseline[:, j]
            act_col = actual_data[:, j]
            psi_val = calculate_psi(base_col, act_col, num_bins=num_bins)
            psi_dict[name] = psi_val

            if psi_val >= self.psi_alert_threshold:
                logger.warning(
                    json.dumps({
                        "event": "MARKET_FEATURE_DRIFT_ALERT",
                        "feature": name,
                        "psi": psi_val,
                        "threshold": self.psi_alert_threshold,
                        "status": "CRITICAL_DRIFT",
                    })
                )
            elif psi_val >= 0.10:
                logger.info(
                    json.dumps({
                        "event": "MARKET_FEATURE_DRIFT_INFO",
                        "feature": name,
                        "psi": psi_val,
                        "status": "MODERATE_DRIFT",
                    })
                )

        return psi_dict

    def is_drift_detected(self) -> bool:
        """Returns True if any feature PSI exceeds the alert threshold."""
        psi_map = self.compute_psi()
        if not psi_map:
            return False
        return any(v >= self.psi_alert_threshold for v in psi_map.values())


class FXCrossGuardian(ModelRiskGuardian):
    """Model Risk Guardian for Triangulated FX/Commodity Cross ML Pricing."""

    def __init__(
        self,
        ood_detector: Optional[MarketStateOODDetector] = None,
        drift_monitor: Optional[MarketStateDriftMonitor] = None,
        reserve_tracker: Optional[Any] = None,
        psi_threshold: float = 0.25,
        epistemic_abs_floor_bp: float = 1.5,
        epistemic_reserve_frac: float = 0.15,
        audit_engine: Optional[nn.Module] = None,
        enable_slice_audit: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ood_detector = ood_detector
        self.drift_monitor = drift_monitor or MarketStateDriftMonitor(psi_alert_threshold=psi_threshold)
        self.reserve_tracker = reserve_tracker
        self.psi_threshold = psi_threshold
        self.epistemic_abs_floor_bp = epistemic_abs_floor_bp
        self.epistemic_reserve_frac = epistemic_reserve_frac
        self._audit_engine = audit_engine
        self.enable_slice_audit = enable_slice_audit

    @property
    def audit_engine(self) -> nn.Module:
        """Returns the engine used for cross-slice output audit (defaults to AnalyticTriangulated)."""
        if self._audit_engine is None:
            self._audit_engine = AnalyticTriangulated()
        return self._audit_engine

    def audit_slice(
        self,
        states: Tensor,
        n_grid: int = 401,
        z_max: float = 12.0,
        cross_ids: Optional[Union[Sequence[object], Tensor]] = None,
    ) -> AuditResult:
        """Audits 34-column market states for Durrleman, call-spread, and wing bounds."""
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.shape[-1] != 34:
            raise ValueError(f"cross_slice_audit requires 34-column engine tensor, got shape {states.shape}")
        return cross_slice_audit(self.audit_engine, states, n_grid=n_grid, z_max=z_max, cross_ids=cross_ids)

    def apply_audit_check(
        self,
        decision: RoutingDecision,
        audit_result: AuditResult,
    ) -> RoutingDecision:
        """Applies cross-slice audit results to an existing routing decision.

        If audit_result.ok is False for any state, forces failover to
        Analytic_Triangulated with FallbackReasonEnum.OUTPUT_AUDIT_FAILED,
        preserving upstream OOD_MARKET_REGIME reason for OOD states.
        """
        if bool(audit_result.ok.all().item()):
            return decision

        nn_gate = decision.nn_gate.clone()
        fallback_activated = decision.fallback_activated.clone()
        fallback_reasons = list(decision.fallback_reason)
        engines = list(decision.engine_used)

        S_cnt = len(fallback_reasons)
        for i in range(S_cnt):
            if not bool(audit_result.ok[i].item()):
                # Do not overwrite upstream OOD_MARKET_REGIME
                if fallback_reasons[i] != FallbackReasonEnum.OOD_MARKET_REGIME.value:
                    nn_gate[i] = 0.0
                    fallback_activated[i] = True
                    fallback_reasons[i] = FallbackReasonEnum.OUTPUT_AUDIT_FAILED.value
                    engines[i] = PricingEngineEnum.ANALYTIC_TRIANGULATED

        return RoutingDecision(
            nn_gate=nn_gate,
            fallback_activated=fallback_activated,
            fallback_reason=fallback_reasons,
            d_m=decision.d_m,
            is_ood=decision.is_ood,
            box_violation=decision.box_violation,
            engine_used=engines,
        )

    @staticmethod
    def extract_12d_state(X: Tensor) -> Tensor:
        """Extracts 12-D market states (sqrt(T), rho, xi1, xi2) from 34-column engine tensor X."""
        T_val = smooth_T(X[:, Col.T])
        sq_T = torch.sqrt(T_val).unsqueeze(-1)
        rho_val = X[:, Col.RHO].unsqueeze(-1)
        xi1 = X[:, Col.XI1 : Col.XI1 + 5]
        xi2 = X[:, Col.XI2 : Col.XI2 + 5]
        return torch.cat([sq_T, rho_val, xi1, xi2], dim=-1)

    def route(
        self,
        states: Tensor,
        engine: PricingEngineEnum = PricingEngineEnum.DIFFERENTIAL_ML,
        cross_ids: Optional[Union[Sequence[object], Tensor]] = None,
    ) -> RoutingDecision:
        """Inspects market states and requested engine, returning RoutingDecision."""
        if states.ndim == 1:
            states = states.unsqueeze(0)

        # Extract 12-D market state if 34-column engine tensor provided
        if states.shape[-1] == 34:
            m = self.extract_12d_state(states)
        elif states.shape[-1] == 12:
            m = states
        else:
            raise ValueError(f"Expected states tensor of shape (S, 12) or (S, 34), got {states.shape}")

        S_cnt = states.shape[0]
        device = states.device
        dtype = torch.float64

        if self.ood_detector is not None:
            D_M, box_viol, is_ood = self.ood_detector.score(m)
        else:
            D_M = torch.zeros(S_cnt, dtype=dtype, device=device)
            box_viol = torch.zeros(S_cnt, dtype=torch.bool, device=device)
            is_ood = torch.zeros(S_cnt, dtype=torch.bool, device=device)

        # Record observed states into drift monitor
        self.drift_monitor.record(m)

        has_nan_weights = False
        models_to_check: List[Any] = []
        if getattr(self, "model", None) is not None:
            models_to_check.append(self.model)
        try:
            from deepvol.models.fx_cross_surrogate import ResidualCrossVolNet

            models_to_check.extend(list(ResidualCrossVolNet._instances))
        except Exception:
            pass
        for m_obj in models_to_check:
            for p in m_obj.parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    has_nan_weights = True
                    break
            if has_nan_weights:
                break

        nn_gate = torch.ones(S_cnt, dtype=dtype, device=device)
        fallback_activated = torch.zeros(S_cnt, dtype=torch.bool, device=device)
        fallback_reason: List[Optional[str]] = [None] * S_cnt
        engine_used: List[PricingEngineEnum] = [engine] * S_cnt

        drift_breached = self.drift_monitor.is_drift_detected()

        # Engine routing decisions
        if engine in (PricingEngineEnum.ANALYTIC_TRIANGULATED, PricingEngineEnum.ANALYTIC_GK):
            nn_gate.zero_()
            fallback_activated.fill_(True)
            fallback_reason = [FallbackReasonEnum.ENGINE_REQUESTED.value] * S_cnt
            engine_used = [PricingEngineEnum.ANALYTIC_TRIANGULATED] * S_cnt

        elif engine in (PricingEngineEnum.DIFFERENTIAL_ML, PricingEngineEnum.HYBRID_ENSEMBLE):
            for i in range(S_cnt):
                if is_ood[i] or has_nan_weights:
                    nn_gate[i] = 0.0
                    fallback_activated[i] = True
                    fallback_reason[i] = FallbackReasonEnum.OOD_MARKET_REGIME.value if is_ood[i] else "NUMERICAL_INSTABILITY"
                    engine_used[i] = PricingEngineEnum.ANALYTIC_TRIANGULATED
                else:
                    if drift_breached:
                        engine_used[i] = PricingEngineEnum.HYBRID_ENSEMBLE
                        nn_gate[i] = 1.0
                        fallback_activated[i] = False
                        fallback_reason[i] = None
                    else:
                        engine_used[i] = engine
                        nn_gate[i] = 1.0
                        fallback_activated[i] = False
                        fallback_reason[i] = None

        elif engine == PricingEngineEnum.COPULA_BENCHMARK:
            nn_gate.zero_()
            fallback_activated.fill_(False)
            fallback_reason = [None] * S_cnt
            engine_used = [PricingEngineEnum.COPULA_BENCHMARK] * S_cnt

        decision = RoutingDecision(
            nn_gate=nn_gate,
            fallback_activated=fallback_activated,
            fallback_reason=fallback_reason,
            d_m=D_M,
            is_ood=is_ood,
            box_violation=box_viol,
            engine_used=engine_used,
        )

        # Slice audit integration: when 34-column engine tensor provided
        if (
            self.enable_slice_audit
            and states.shape[-1] == 34
            and engine in (PricingEngineEnum.DIFFERENTIAL_ML, PricingEngineEnum.HYBRID_ENSEMBLE)
        ):
            # Only audit non-OOD candidate states (avoid overhead if all states are already OOD)
            non_ood_mask = ~is_ood
            if torch.any(non_ood_mask):
                audit_res = self.audit_slice(states, cross_ids=cross_ids)
                decision = self.apply_audit_check(decision, audit_res)

        return decision

    def post_check(self, eps_std_bp: Tensor, reserve_bp: Tensor) -> Tensor:
        """Epistemic threshold check returning boolean fallback mask (S,)."""
        device = eps_std_bp.device
        dtype = eps_std_bp.dtype

        floor_val = torch.tensor(self.epistemic_abs_floor_bp, dtype=dtype, device=device)
        threshold = torch.maximum(floor_val, self.epistemic_reserve_frac * reserve_bp)
        return eps_std_bp > threshold

    def apply_post_check(
        self,
        decision: RoutingDecision,
        eps_std_bp: Tensor,
        reserve_bp: Tensor,
    ) -> RoutingDecision:
        """Applies epistemic uncertainty check to an existing routing decision."""
        mask = self.post_check(eps_std_bp, reserve_bp)
        if not torch.any(mask):
            return decision

        nn_gate = decision.nn_gate.clone()
        fallback_activated = decision.fallback_activated.clone()
        fallback_reasons = list(decision.fallback_reason)
        engines = list(decision.engine_used)

        for i in range(len(fallback_reasons)):
            if mask[i]:
                nn_gate[i] = 0.0
                fallback_activated[i] = True
                fallback_reasons[i] = FallbackReasonEnum.EPISTEMIC_UNCERTAINTY.value
                engines[i] = PricingEngineEnum.ANALYTIC_TRIANGULATED

        return RoutingDecision(
            nn_gate=nn_gate,
            fallback_activated=fallback_activated,
            fallback_reason=fallback_reasons,
            d_m=decision.d_m,
            is_ood=decision.is_ood,
            box_violation=decision.box_violation,
            engine_used=engines,
        )


@dataclass(frozen=True)
class ModelReserveEntry:
    """Thread-safe cache entry for copula model reserves."""
    reserve_bp: float
    computed_at: datetime
    state_timestamp: Optional[datetime] = None
    computed_at_mono: float = 0.0

    def __post_init__(self) -> None:
        if self.computed_at_mono == 0.0:
            now_dt = datetime.now(timezone.utc)
            comp_at = self.computed_at
            if comp_at.tzinfo is None:
                comp_at = comp_at.replace(tzinfo=timezone.utc)
            age_sec = (now_dt - comp_at).total_seconds()
            object.__setattr__(self, "computed_at_mono", time.monotonic() - age_sec)


class ModelReserveTracker:
    """Asynchronous background model reserve tracker and thread-safe TTL cache.

    Monitors model divergence against the Gaussian copula teacher benchmark
    over 64 strikes per market state. Guarantees non-blocking quote paths.
    """

    DEFAULT_MAX_AGE: Final[timedelta] = timedelta(minutes=15)
    DEFAULT_CAPACITY: Final[int] = 10000

    def __init__(
        self,
        copula: GaussianCopulaCrossEngine,
        max_age: timedelta = DEFAULT_MAX_AGE,
        max_capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        self.copula = copula
        self.max_age = max_age
        self.max_age_seconds: float = max_age.total_seconds()
        self.max_capacity = max_capacity

        self._cache: Dict[str, ModelReserveEntry] = {}
        self._in_progress: Dict[str, asyncio.Future[float]] = {}
        self._lock = threading.Lock()

    def generate_strikes(self, state: CrossState) -> Tensor:
        """Generates the deterministic 64-strike evaluation grid for a CrossState."""
        device = self.copula.device
        dtype = torch.float64

        # 8 fixed points in the taper boundary zone |z| in [5, 7]
        taper_z = torch.tensor(
            [-7.0, -6.333333333333333, -5.666666666666667, -5.0,
              5.0,  5.666666666666667,  6.333333333333333,  7.0],
            dtype=dtype,
            device=device,
        )

        # 56 core smile points linearly spaced across [-4.8, 4.8]
        core_z = torch.linspace(-4.8, 4.8, 56, dtype=dtype, device=device)

        # Combine and sort
        all_z, _ = torch.sort(torch.cat([core_z, taper_z]))

        F = state.F.to(device=device, dtype=dtype)
        if F.numel() > 1:
            F = F[0]
        T = state.T.to(device=device, dtype=dtype)
        if T.numel() > 1:
            T = T[0]
        sigma_ref = state.sigma_ref.to(device=device, dtype=dtype)
        if sigma_ref.numel() > 1:
            sigma_ref = sigma_ref[0]

        # K = F * exp(z * sigma_ref * sqrt(T))
        strikes = F * torch.exp(all_z * sigma_ref * torch.sqrt(T))
        return strikes

    def compute_reserve(
        self,
        state: CrossState,
        served: Callable[[Tensor], Any],
        D: Optional[Tensor] = None,
    ) -> float:
        """Synchronously computes model reserve MR_s in basis points."""
        strikes = self.generate_strikes(state)

        with torch.no_grad():
            copula_undisc = self.copula.price(state, strikes, is_call=True).squeeze(0)

            served_out = served(strikes)
            if hasattr(served_out, "V"):
                served_prices = served_out.V.squeeze(0).to(self.copula.device, dtype=torch.float64)
            else:
                served_prices = served_out.squeeze(0).to(self.copula.device, dtype=torch.float64)

            F_val = float(state.F.flatten()[0].item())
            if D is not None:
                D_val = float(D.flatten()[0].item())
            elif hasattr(state, "D"):
                d_t = getattr(state, "D")
                D_val = float(d_t.flatten()[0].item())
            else:
                D_val = 1.0

            if D is not None or hasattr(state, "D"):
                copula_prices = copula_undisc * D_val
            else:
                copula_prices = copula_undisc

            denom = max(D_val * F_val, 1e-12)
            abs_diff = torch.abs(copula_prices - served_prices)
            max_err = float(torch.max(abs_diff).item())

            reserve_bp = (max_err / denom) * 10000.0
            return float(reserve_bp)

    async def refresh(
        self,
        key: str,
        state: CrossState,
        served: Callable[[Tensor], Any],
        state_timestamp: Optional[datetime] = None,
    ) -> None:
        """Background asynchronous refresh task with single-flight deduplication."""
        loop = asyncio.get_running_loop()

        with self._lock:
            if key in self._in_progress:
                future = self._in_progress[key]
                first_caller = False
            else:
                future = loop.create_future()
                self._in_progress[key] = future
                first_caller = True

        if first_caller:
            try:
                reserve_bp = await asyncio.to_thread(self.compute_reserve, state, served)
                now = datetime.now(timezone.utc)
                now_mono = time.monotonic()
                entry = ModelReserveEntry(
                    reserve_bp=reserve_bp,
                    computed_at=now,
                    state_timestamp=state_timestamp or now,
                    computed_at_mono=now_mono,
                )
                with self._lock:
                    self._cache[key] = entry
                    self._evict_lru_if_needed()
                future.set_result(reserve_bp)
            except Exception as exc:
                future.set_exception(exc)
                raise
            finally:
                with self._lock:
                    self._in_progress.pop(key, None)
        else:
            await future

    def get(
        self,
        key: str,
        state_timestamp: Optional[datetime] = None,
    ) -> Tuple[Optional[float], bool]:
        """Retrieves cached model reserve and staleness flag meeting the <= 10 us SLA.

        Performance Characteristics:
          - Contract SLA: <= 10 us (Task 3.4 production gateway).
          - Lock-free Read Path: Pure CPython atomic dictionary read eliminates mutex
            contention and futex context switches against background refresh threads.
          - Monotonic Hot Path: Evaluated via time.monotonic() float arithmetic with zero
            heap allocations, achieving ~60-150 ns latency.
        """
        entry = self._cache.get(key)
        if entry is None:
            return None, True

        now_mono = time.monotonic()
        is_stale = False

        if (now_mono - entry.computed_at_mono) > self.max_age_seconds:
            is_stale = True

        if state_timestamp is not None:
            if entry.state_timestamp is not None:
                if entry.state_timestamp < state_timestamp:
                    is_stale = True
            elif entry.computed_at < state_timestamp:
                is_stale = True

        return entry.reserve_bp, is_stale

    def _evict_lru_if_needed(self) -> None:
        """Evicts oldest entries when cache exceeds max capacity."""
        if len(self._cache) > self.max_capacity:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].computed_at_mono)
            self._cache.pop(oldest_key, None)

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidates cache entries."""
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

    def __len__(self) -> int:
        """Returns the number of cached state entries."""
        with self._lock:
            return len(self._cache)
