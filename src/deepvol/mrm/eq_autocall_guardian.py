"""eq_autocall_guardian.py — Tier 3 SR 11-7 / SR 26-2 Model Risk Management Guardian & Failover Router.

Implements SPEC-DEEPVOL-EQ-AUTOCALL-001 §1.5, Milestone M4 (Task 4.1), and Gates G3.1, G3.2, G3.4
in strict IEEE-754 double precision (`np.float64` / `torch.float64`):
  - `AutocallFallbackReason`: Normative 13-reason fallback enumeration for structural, domain,
    correlation, pin-risk, OOD/drift, and post-inference sanity tripwires.
  - `SR262AuditRecord`: Structured Federal Reserve SR 11-7 / SR 26-2 compliance telemetry record.
  - `project_correlation_to_psd`: Ledoit-Wolf convex shrinkage + spectral eigenvalue floor ensuring
    lambda_min(R_PSD) >= 1e-4 and exact unit diagonal so downstream Cholesky never fails.
  - `extract_16d_basket_summary`: Canonical 16D basket summary feature vector extractor for
    Mahalanobis OOD screening (D_M^2 <= chi2_{16, 0.999} = 39.25) and 10-decile rolling PSI drift
    monitoring (warning at PSI >= 0.10, fallback at PSI >= 0.25).
  - `EQAutocallModelGuardian`: 3-stage Pre-Inference / Post-Inference circuit breaker and
    deterministic failover router to `price_eq_autocall_dupire_qmc`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
import hashlib
import logging
import math
from pathlib import Path
import threading
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
import torch

from deepvol.models.autocall.analytical_baseline import (
    AutocallBaselineOutput,
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.dupire_qmc_solver import price_eq_autocall_dupire_qmc
from deepvol.models.autocall.dupire_qmc_solver import DupireQMCResult
from deepvol.models.autocall.feature_layout import (
    D_MAX_DEFAULT,
    M_MAX_DEFAULT,
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
    MonitoringIndexCode,
    monitoring_index_to_code,
)
from deepvol.models.autocall.pre_post_processor import (
    AutocallPostprocOutput,
    AutocallPreprocState,
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.seasoned_collapser import (
    CashFlowItem,
    evaluate_monitoring_index_np,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
)

logger = logging.getLogger("deepvol.mrm.eq_autocall_guardian")

DEFAULT_GUARDIAN_STATS_PATH: Path = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "checkpoints"
    / "eq_autocall_guardian_stats.pt"
)

SUPPORTED_TIER2_INDICES: frozenset[str] = frozenset(
    {"WorstOf", "BestOf", "Average", "AverageOf"}
)

CHI2_16_0999: float = 39.25
DEFAULT_TAU_DESK_BPS: float = 60.0


class AutocallFallbackReason(str, Enum):
    """Reason codes for EQAutocallModelGuardian circuit breaker and failover routing."""

    BASKET_CARDINALITY_EXCEEDED = "BASKET_CARDINALITY_EXCEEDED"
    SCHEDULE_HORIZON_EXCEEDED = "SCHEDULE_HORIZON_EXCEEDED"
    ZERO_FUTURE_PERIODS = "ZERO_FUTURE_PERIODS"
    EXOTIC_MONITORING_INDEX = "EXOTIC_MONITORING_INDEX"
    EXOTIC_TERMINAL_MODIFIER_ROUTED = "EXOTIC_TERMINAL_MODIFIER_ROUTED"
    CORRELATION_NOT_PSD = "CORRELATION_NOT_PSD"
    IMMINENT_BARRIER_PIN_RISK = "IMMINENT_BARRIER_PIN_RISK"
    MAHALANOBIS_OOD = "MAHALANOBIS_OOD"
    CONFORMAL_UNCERTAINTY_BREACH = "CONFORMAL_UNCERTAINTY_BREACH"
    PSI_DRIFT_BREACH = "PSI_DRIFT_BREACH"
    POST_INFERENCE_SIMPLEX_VIOLATION = "POST_INFERENCE_SIMPLEX_VIOLATION"
    POST_INFERENCE_PUT_BOUND_VIOLATION = "POST_INFERENCE_PUT_BOUND_VIOLATION"
    POST_INFERENCE_RESIDUAL_OUTLIER = "POST_INFERENCE_RESIDUAL_OUTLIER"
    NON_FINITE_OUTPUT = "NON_FINITE_OUTPUT"
    ENGINE_REQUESTED_QMC = "ENGINE_REQUESTED_QMC"


class AutocallSummaryCol(IntEnum):
    """16D basket summary feature layout for Mahalanobis OOD & 10-decile PSI monitoring."""

    MEAN_LOG_MONEYNESS = 0
    MIN_LOG_MONEYNESS = 1
    MAX_LOG_MONEYNESS = 2
    MEAN_ATM_VOL = 3
    MAX_ATM_VOL = 4
    MEAN_CORR = 5
    MIN_CORR = 6
    MAX_CASH_DIV_LEVERAGE = 7
    TERM_MATURITY = 8
    LOG_KO_BARRIER = 9
    LOG_CPN_BARRIER = 10
    PUT_STRIKE = 11
    STULZ_LOG_FWD_W = 12
    STULZ_VOL_W = 13
    MEMORY_COUPON_BUFFER = 14
    ABS_DIGITAL_SPREAD = 15


FEATURE_NAMES_16D: tuple[str, ...] = (
    "mean_log_moneyness",
    "min_log_moneyness",
    "max_log_moneyness",
    "mean_atm_vol",
    "max_atm_vol",
    "mean_corr",
    "min_corr",
    "max_cash_div_leverage",
    "term_maturity",
    "log_ko_barrier",
    "log_cpn_barrier",
    "put_strike",
    "stulz_log_fwd_w",
    "stulz_vol_w",
    "memory_coupon_buffer",
    "abs_digital_spread",
)

# Calibrated reference mean and standard deviation for the 16D summary feature space
_DEFAULT_REF_MEAN_16D: npt.NDArray[np.float64] = np.array(
    [
        -0.20,  # 0: mean_log_moneyness
        -0.50,  # 1: min_log_moneyness
        0.05,   # 2: max_log_moneyness
        0.35,   # 3: mean_atm_vol
        0.45,   # 4: max_atm_vol
        0.45,   # 5: mean_corr
        0.25,   # 6: min_corr
        0.08,   # 7: max_cash_div_leverage
        1.00,   # 8: term_maturity
        0.05,   # 9: log_ko_barrier
        -0.50,  # 10: log_cpn_barrier
        0.85,   # 11: put_strike
        -0.60,  # 12: stulz_log_fwd_w
        0.25,   # 13: stulz_vol_w
        0.10,   # 14: memory_coupon_buffer
        0.10,   # 15: abs_digital_spread
    ],
    dtype=np.float64,
)

_DEFAULT_REF_STD_16D: npt.NDArray[np.float64] = np.array(
    [
        0.38,   # 0: mean_log_moneyness
        0.42,   # 1: min_log_moneyness
        0.38,   # 2: max_log_moneyness
        0.11,   # 3: mean_atm_vol
        0.13,   # 4: max_atm_vol
        0.32,   # 5: mean_corr
        0.40,   # 6: min_corr
        0.045,  # 7: max_cash_div_leverage
        1.10,   # 8: term_maturity
        0.45,   # 9: log_ko_barrier
        0.35,   # 10: log_cpn_barrier
        0.30,   # 11: put_strike
        0.50,   # 12: stulz_log_fwd_w
        0.14,   # 13: stulz_vol_w
        0.22,   # 14: memory_coupon_buffer
        0.12,   # 15: abs_digital_spread
    ],
    dtype=np.float64,
)


@dataclass
class SR262AuditRecord:
    """Federal Reserve SR 11-7 / SR 26-2 compliance audit log record."""

    trade_id: str
    timestamp_utc: str
    pre_gate_passed: bool
    post_gate_passed: bool
    fallback_triggered: bool
    fallback_reasons: list[str]
    warnings: list[str]
    min_corr_eigenvalue_raw: float
    min_corr_eigenvalue_shrunk: float
    correlation_shrunk_applied: bool
    mahalanobis_d2: float
    max_psi_score: float
    min_pin_risk_z: float
    engine_selected: str
    conformal_half_width_bps: float = 0.0
    greeks_engine_used: str = "None"

    def to_dict(self) -> dict[str, Any]:
        """Serializes the audit record to a JSON-compatible dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp_utc": self.timestamp_utc,
            "pre_gate_passed": self.pre_gate_passed,
            "post_gate_passed": self.post_gate_passed,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reasons": list(self.fallback_reasons),
            "warnings": list(self.warnings),
            "min_corr_eigenvalue_raw": self.min_corr_eigenvalue_raw,
            "min_corr_eigenvalue_shrunk": self.min_corr_eigenvalue_shrunk,
            "correlation_shrunk_applied": self.correlation_shrunk_applied,
            "mahalanobis_d2": self.mahalanobis_d2,
            "max_psi_score": self.max_psi_score,
            "min_pin_risk_z": self.min_pin_risk_z,
            "engine_selected": self.engine_selected,
            "conformal_half_width_bps": self.conformal_half_width_bps,
            "greeks_engine_used": self.greeks_engine_used,
        }


@dataclass
class GuardianPricingResult:
    """Complete guarded valuation result emitted by EQAutocallModelGuardian."""

    product_value: float
    standard_error: float
    notional_pv: float
    coupon_pv: float
    option_pv: float
    funding_pv: float
    pv_in_transit: float
    p_stop: npt.NDArray[np.float64]
    q_cpn: npt.NDArray[np.float64]
    l_put: float
    l_call: float
    cashflows: list[CashFlowItem]
    reporting_currency: str
    cash_to_settle_today: float
    engine_selected: str
    fallback_triggered: bool
    fallback_reasons: list[AutocallFallbackReason]
    audit_record: SR262AuditRecord
    shrunk_corr_matrix: npt.NDArray[np.float64]
    preproc_state: AutocallPreprocState
    postproc_out: AutocallPostprocOutput | None = None
    qmc_result: DupireQMCResult | None = None
    surrogate_out: AutocallSurrogateOutput | None = None
    greeks_out: Any = None

    @property
    def ProductValue(self) -> float:
        return self.product_value

    @property
    def StandardError(self) -> float:
        return self.standard_error

    @property
    def Option(self) -> float:
        return self.option_pv

    @property
    def Notional(self) -> float:
        return self.notional_pv

    @property
    def Coupon(self) -> float:
        return self.coupon_pv

    @property
    def CashFlows(self) -> list[CashFlowItem]:
        return self.cashflows

    @property
    def cash_flows(self) -> list[CashFlowItem]:
        return self.cashflows

    def to_dict(self) -> dict[str, Any]:
        """Serializes the guarded pricing result into a Murex-compatible response dict."""
        return {
            "ProductValue": float(self.product_value),
            "StandardError": float(self.standard_error),
            "Notional": float(self.notional_pv),
            "Coupon": float(self.coupon_pv),
            "Option": float(self.option_pv),
            "FundingLeg": float(-self.funding_pv),
            "ReportingCurrency": self.reporting_currency,
            "CashToSettleToday": float(self.cash_to_settle_today),
            "CashFlows": [cf.to_dict() for cf in self.cashflows],
            "EngineSelected": self.engine_selected,
            "FallbackTriggered": bool(self.fallback_triggered),
            "FallbackReasons": [r.value for r in self.fallback_reasons],
            "AuditRecord": self.audit_record.to_dict(),
        }


def project_correlation_to_psd(
    corr: npt.NDArray[np.float64] | torch.Tensor | Sequence[Sequence[float]],
    *,
    eps_psd: float = 1e-4,
) -> tuple[npt.NDArray[np.float64], float, float, bool]:
    """Projects a correlation matrix R onto {S >= eps_psd I} \\cap {diag(S) = 1} via Dykstra-Higham (2002).

    Returns:
        r_psd: (D, D) float64 PSD correlation matrix with unit diagonal and lambda_min >= eps_psd.
        min_eig_raw: Minimum eigenvalue of the raw symmetrized input matrix.
        min_eig_shrunk: Minimum eigenvalue of the returned R_PSD matrix.
        shrunk_applied: True if asymmetry, diagonal deviation, or non-PSD projection occurred.
    """
    if isinstance(corr, torch.Tensor):
        r_in = corr.detach().cpu().numpy().astype(np.float64, copy=True)
    else:
        r_in = np.asarray(corr, dtype=np.float64).copy()

    if r_in.ndim != 2 or r_in.shape[0] != r_in.shape[1]:
        raise ValueError(f"Expected square 2D correlation matrix, got shape {r_in.shape}.")

    d = r_in.shape[0]
    if d == 0:
        return np.zeros((0, 0), dtype=np.float64), 1.0, 1.0, False
    if d == 1:
        raw_val = float(r_in[0, 0]) if np.isfinite(r_in[0, 0]) else -1.0
        shrunk = abs(raw_val - 1.0) > 1e-12
        return np.ones((1, 1), dtype=np.float64), raw_val, 1.0, shrunk

    has_non_finite = not bool(np.all(np.isfinite(r_in)))
    if has_non_finite:
        r_clean = np.where(np.isfinite(r_in), r_in, 0.0)
        np.fill_diagonal(r_clean, 1.0)
        has_asymmetry = float(np.max(np.abs(r_clean - r_clean.T))) > 1e-10
        r_sym = 0.5 * (r_clean + r_clean.T)
        min_eig_raw = -1.0
    else:
        has_asymmetry = float(np.max(np.abs(r_in - r_in.T))) > 1e-10
        r_sym = 0.5 * (r_in + r_in.T)
        eigvals_raw = np.linalg.eigvalsh(r_sym)
        min_eig_raw = float(np.min(eigvals_raw))

    diag_dev = float(np.max(np.abs(np.diag(r_sym) - 1.0)))
    off_mask = ~np.eye(d, dtype=bool)
    max_off = float(np.max(np.abs(r_sym[off_mask]))) if np.any(off_mask) else 0.0

    if (
        not has_non_finite
        and not has_asymmetry
        and min_eig_raw >= eps_psd
        and diag_dev <= 1e-10
        and max_off <= 1.0 - 1e-12
    ):
        np.fill_diagonal(r_sym, 1.0)
        return r_sym, min_eig_raw, min_eig_raw, False

    # Dykstra-Higham (2002) alternating projections onto {S >= eps_psd * I} \cap {diag(S) = 1}
    y = r_sym.copy()
    np.fill_diagonal(y, 1.0)
    d_s = np.zeros_like(y)
    for _ in range(50):
        r_k = y - d_s
        eigvals, eigvecs = np.linalg.eigh(0.5 * (r_k + r_k.T))
        eigvals_clipped = np.maximum(eigvals, eps_psd)
        x_k = (eigvecs * eigvals_clipped[None, :]) @ eigvecs.T
        x_k = 0.5 * (x_k + x_k.T)
        d_s = x_k - r_k
        y_next = x_k.copy()
        np.fill_diagonal(y_next, 1.0)
        step_diff = float(np.max(np.abs(y_next - y)))
        y = y_next
        if step_diff <= 1e-12:
            if float(np.min(np.linalg.eigvalsh(y))) >= eps_psd - 1e-12:
                break

    y = 0.5 * (y + y.T)
    np.fill_diagonal(y, 1.0)
    min_eig_shrunk = float(np.min(np.linalg.eigvalsh(y)))
    if min_eig_shrunk < eps_psd:
        alpha_fix = min(
            max((eps_psd - min_eig_shrunk + 1e-12) / max(1.0 - min_eig_shrunk, 1e-12), 0.0),
            1.0,
        )
        y = (1.0 - alpha_fix) * y + alpha_fix * np.eye(d, dtype=np.float64)
        y = 0.5 * (y + y.T)
        np.fill_diagonal(y, 1.0)
        min_eig_shrunk = float(np.min(np.linalg.eigvalsh(y)))

    return y, min_eig_raw, min_eig_shrunk, True


def extract_16d_basket_summary(
    preproc_state: AutocallPreprocState,
    baseline_out: AutocallBaselineOutput | None = None,
) -> npt.NDArray[np.float64]:
    """Extracts the canonical 16D basket summary vector x in R^16 in IEEE-754 float64.

    Coordinates (see `AutocallSummaryCol` / `FEATURE_NAMES_16D`):
      0: mean_log_moneyness
      1: min_log_moneyness
      2: max_log_moneyness
      3: mean_atm_vol
      4: max_atm_vol
      5: mean_corr
      6: min_corr
      7: max_cash_div_leverage
      8: term_maturity
      9: log_ko_barrier
     10: log_cpn_barrier
     11: put_strike
     12: stulz_log_fwd_w
     13: stulz_vol_w
     14: memory_coupon_buffer
     15: abs_digital_spread
    """
    d = int(preproc_state.d_assets)
    m_rem = int(preproc_state.m_rem)
    b_out = baseline_out if baseline_out is not None else preproc_state.baseline

    x = np.zeros(16, dtype=np.float64)

    if d > 0:
        m0_np = preproc_state.moneyness_t0.detach().cpu().numpy().astype(np.float64)
        log_m0 = np.log(np.maximum(m0_np, 1e-12))
        x[AutocallSummaryCol.MEAN_LOG_MONEYNESS] = float(np.mean(log_m0))
        x[AutocallSummaryCol.MIN_LOG_MONEYNESS] = float(np.min(log_m0))
        x[AutocallSummaryCol.MAX_LOG_MONEYNESS] = float(np.max(log_m0))

        node_np = (
            preproc_state.node_feats[0, :d, :].detach().cpu().numpy().astype(np.float64)
        )
        iv_short = node_np[:, int(AutocallNodeCol.IV_SHORT_ATM)]
        iv_term = node_np[:, int(AutocallNodeCol.IV_TERM_ATM)]
        x[AutocallSummaryCol.MEAN_ATM_VOL] = float(np.mean(iv_term))
        x[AutocallSummaryCol.MAX_ATM_VOL] = float(np.max(np.maximum(iv_short, iv_term)))
        x[AutocallSummaryCol.MAX_CASH_DIV_LEVERAGE] = float(
            np.max(node_np[:, int(AutocallNodeCol.CASH_DIV_LEVERAGE)])
        )
    else:
        log_m0 = np.zeros(1, dtype=np.float64)

    corr_np = preproc_state.corr_matrix.detach().cpu().numpy().astype(np.float64)
    if d >= 2 and corr_np.shape == (d, d):
        off_vals = corr_np[~np.eye(d, dtype=bool)]
        x[AutocallSummaryCol.MEAN_CORR] = float(np.mean(off_vals))
        x[AutocallSummaryCol.MIN_CORR] = float(np.min(off_vals))
    else:
        x[AutocallSummaryCol.MEAN_CORR] = 0.45
        x[AutocallSummaryCol.MIN_CORR] = 0.25

    if m_rem > 0:
        tau_np = preproc_state.tau_obs.detach().cpu().numpy().astype(np.float64)
        x[AutocallSummaryCol.TERM_MATURITY] = float(tau_np[-1])

        sched_np = (
            preproc_state.sched_feats[0, :m_rem, :]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        call_mask = sched_np[:, int(AutocallSchedCol.IS_CALLABLE_FLAG)] > 0.5
        if np.any(call_mask):
            x[AutocallSummaryCol.LOG_KO_BARRIER] = float(
                np.mean(sched_np[call_mask, int(AutocallSchedCol.LOG_KO_BARRIER)])
            )
        else:
            x[AutocallSummaryCol.LOG_KO_BARRIER] = 0.05
        log_cpn_col = sched_np[:, int(AutocallSchedCol.LOG_CPN_BARRIER)]
        guar_col = sched_np[:, int(AutocallSchedCol.IS_GUARANTEED_CPN)]
        cpn_active_mask = (guar_col < 0.5) & (log_cpn_col > math.log(1e-4) + 1e-4)
        if np.any(cpn_active_mask):
            x[AutocallSummaryCol.LOG_CPN_BARRIER] = float(np.mean(log_cpn_col[cpn_active_mask]))
        else:
            x[AutocallSummaryCol.LOG_CPN_BARRIER] = -0.50
    else:
        x[AutocallSummaryCol.TERM_MATURITY] = 0.0
        x[AutocallSummaryCol.LOG_KO_BARRIER] = 0.05
        x[AutocallSummaryCol.LOG_CPN_BARRIER] = -0.50

    x[AutocallSummaryCol.PUT_STRIKE] = float(preproc_state.strike)

    if b_out is not None and m_rem > 0 and b_out.f_w.numel() > 0:
        fw_val = float(b_out.f_w[-1].detach().cpu().item())
        vw_val = float(b_out.v_w[-1].detach().cpu().item())
        x[AutocallSummaryCol.STULZ_LOG_FWD_W] = (
            math.log(max(fw_val, 1e-12)) if np.isfinite(fw_val) else float("nan")
        )
        x[AutocallSummaryCol.STULZ_VOL_W] = vw_val
    else:
        x[AutocallSummaryCol.STULZ_LOG_FWD_W] = float(np.min(log_m0))
        x[AutocallSummaryCol.STULZ_VOL_W] = 0.25

    x[AutocallSummaryCol.MEMORY_COUPON_BUFFER] = float(preproc_state.c_mem_0)
    x[AutocallSummaryCol.ABS_DIGITAL_SPREAD] = abs(float(preproc_state.ko_spread))
    return x


def compute_psi_10_decile(
    observed_samples: npt.NDArray[np.float64],
    bin_edges: npt.NDArray[np.float64],
    ref_probs: npt.NDArray[np.float64],
) -> tuple[float, dict[str, float]]:
    """Computes the Laplace-smoothed 10-decile Population Stability Index (PSI) across the 16 summary features.

    Formula (with Laplace pseudocount alpha_0 = 0.5, K = 10 bins):
        p_hat_{j,b} = (c_{j,b} + 0.5) / (N + 5.0)
        q_hat_{j,b} = (N * q_{j,b} + 0.5) / (N + 5.0)
        PSI_j = sum_{b=1}^{10} (p_hat_{j,b} - q_hat_{j,b}) * ln(p_hat_{j,b} / q_hat_{j,b})

    Returns:
        max_psi: Maximum PSI across all 16 features.
        psi_by_feature: Dictionary mapping feature name -> PSI score.
    """
    obs = np.asarray(observed_samples, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    n_samples, n_feats = obs.shape
    if n_samples == 0:
        return 0.0, {name: 0.0 for name in FEATURE_NAMES_16D}

    psi_map: dict[str, float] = {}
    max_psi = 0.0
    for j in range(min(n_feats, len(FEATURE_NAMES_16D))):
        col = obs[:, j]
        col_finite = col[np.isfinite(col)]
        if len(col_finite) == 0:
            psi_j = 10.0
        else:
            n_obs = float(len(col_finite))
            counts, _ = np.histogram(col_finite, bins=bin_edges[j])
            p_b = (counts.astype(np.float64) + 0.5) / (n_obs + 5.0)
            q_raw = np.asarray(ref_probs[j], dtype=np.float64)
            q_b = (n_obs * q_raw + 0.5) / (n_obs + 5.0)
            p_safe = np.maximum(p_b, 1e-12)
            q_safe = np.maximum(q_b, 1e-12)
            psi_j = float(np.sum((p_b - q_b) * np.log(p_safe / q_safe)))
        name_j = FEATURE_NAMES_16D[j]
        psi_map[name_j] = psi_j
        if psi_j > max_psi:
            max_psi = psi_j

    return max_psi, psi_map


_D1_ACTIVE_SUBSPACE_INDICES: tuple[int, ...] = (0, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def _regularize_cov_oas_correlation_space(
    cov_mat: npt.NDArray[np.float64],
    *,
    lambda_min: float = 0.025,
    std_floor: float = 1e-2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Standardizes covariance in correlation space with eigenvalue floor (kappa <= 40) and builds d=1 subspace precision."""
    cov_sym = 0.5 * (cov_mat + cov_mat.T)
    raw_std = np.sqrt(np.maximum(np.diag(cov_sym), 1e-12))
    marg_std = np.maximum(raw_std, float(std_floor))
    r_mat = cov_sym / np.maximum(np.outer(raw_std, raw_std), 1e-12)
    r_sym = 0.5 * (r_mat + r_mat.T)
    np.fill_diagonal(r_sym, 1.0)
    eigvals, eigvecs = np.linalg.eigh(r_sym)
    eigvals_floored = np.maximum(eigvals, float(lambda_min))
    r_reg = (eigvecs * eigvals_floored[None, :]) @ eigvecs.T
    r_inv = (eigvecs * (1.0 / eigvals_floored)[None, :]) @ eigvecs.T
    inv_std = 1.0 / marg_std
    cov_reg = (marg_std[:, None] * r_reg) * marg_std[None, :]
    cov_inv_reg = (inv_std[:, None] * r_inv) * inv_std[None, :]

    sub_idx = list(_D1_ACTIVE_SUBSPACE_INDICES)
    cov_sub = cov_reg[np.ix_(sub_idx, sub_idx)]
    cov_inv_d1 = np.zeros((16, 16), dtype=np.float64)
    cov_inv_d1[np.ix_(sub_idx, sub_idx)] = np.linalg.pinv(cov_sub)
    return cov_reg, cov_inv_reg, marg_std, cov_inv_d1


class EQAutocallModelGuardian:
    """Tier 3 SR 11-7 / SR 26-2 Model Risk Management Guardian & Failover Router.

    Enforces a 3-stage circuit breaker:
      1. Pre-Inference Structural, PSD Correlation, Imminent Pin-Risk, Mahalanobis OOD,
         Conformal Uncertainty & PSI Gate.
      2. Post-Inference Probability Simplex, Put-Spread Bound & Funded Residual Gate.
      3. Deterministic Failover Router to `price_eq_autocall_dupire_qmc` with SR 26-2 Audit Logging.
    """

    def __init__(
        self,
        *,
        d_max: int = D_MAX_DEFAULT,
        m_max: int = M_MAX_DEFAULT,
        eps_psd: float = 1e-4,
        severe_psd_threshold: float = -1e-2,
        pin_risk_tau_max: float = 0.08,
        pin_risk_z_min: float = 0.15,
        mahalanobis_chi2_threshold: float = CHI2_16_0999,
        psi_warning_threshold: float = 0.10,
        psi_fallback_threshold: float = 0.25,
        psi_window_size: int = 200,
        psi_min_samples: int = 20,
        residual_threshold: float | None = None,
        ref_mean_16d: npt.NDArray[np.float64] | None = None,
        ref_cov_inv_16d: npt.NDArray[np.float64] | None = None,
        record_drift_by_default: bool = True,
        psi_hard_circuit_breaker: bool = True,
        tau_desk_bps: float = DEFAULT_TAU_DESK_BPS,
        conformal_alpha: float = 0.05,
        stats_path: str | Path | None = None,
        use_gpu_qmc: bool | None = None,
    ) -> None:
        self.d_max = int(d_max)
        self.m_max = int(m_max)
        self.eps_psd = float(eps_psd)
        self.severe_psd_threshold = float(severe_psd_threshold)
        self.pin_risk_tau_max = float(pin_risk_tau_max)
        self.pin_risk_z_min = float(pin_risk_z_min)
        self.mahalanobis_chi2_threshold = float(mahalanobis_chi2_threshold)
        self.psi_warning_threshold = float(psi_warning_threshold)
        self.psi_fallback_threshold = float(psi_fallback_threshold)
        self.psi_window_size = int(psi_window_size)
        self.psi_min_samples = int(psi_min_samples)
        self.residual_threshold: float | None = (
            float(residual_threshold) if residual_threshold is not None else None
        )
        self.record_drift_by_default = bool(record_drift_by_default)
        self.psi_hard_circuit_breaker = bool(psi_hard_circuit_breaker)
        self.tau_desk_bps = float(tau_desk_bps)
        self.conformal_alpha = float(conformal_alpha)
        self.use_gpu_qmc = use_gpu_qmc

        self.ref_mean_16d: npt.NDArray[np.float64] = (
            np.asarray(ref_mean_16d, dtype=np.float64).copy()
            if ref_mean_16d is not None
            else _DEFAULT_REF_MEAN_16D.copy()
        )
        if ref_cov_inv_16d is not None:
            self.ref_cov_inv_16d: npt.NDArray[np.float64] = np.asarray(
                ref_cov_inv_16d, dtype=np.float64
            ).copy()
            cov_mat = np.linalg.pinv(self.ref_cov_inv_16d)
            _, _, ref_std, cov_inv_d1 = _regularize_cov_oas_correlation_space(cov_mat)
        else:
            ref_std = _DEFAULT_REF_STD_16D.copy()
            self.ref_cov_inv_16d = np.diag(1.0 / (ref_std * ref_std))
            _, _, _, cov_inv_d1 = _regularize_cov_oas_correlation_space(np.diag(ref_std * ref_std))
        self.ref_cov_inv_d1_16d: npt.NDArray[np.float64] = cov_inv_d1
        self.ref_marginal_std_16d: npt.NDArray[np.float64] = ref_std.copy()

        # Initialize 10-decile bin edges from the calibrated Gaussian reference distribution
        self.psi_bin_edges = np.zeros((16, 11), dtype=np.float64)
        self.psi_ref_probs = np.full((16, 10), 0.10, dtype=np.float64)
        inner_z = np.asarray(norm.ppf(np.linspace(0.1, 0.9, 9)), dtype=np.float64)
        for j in range(16):
            self.psi_bin_edges[j, 0] = -np.inf
            self.psi_bin_edges[j, 1:10] = self.ref_mean_16d[j] + ref_std[j] * inner_z
            self.psi_bin_edges[j, 10] = np.inf

        self.q_ood_conformal: float = self.mahalanobis_chi2_threshold
        self.q_err_conformal: float = 0.0
        self.scale_eps0: float = 0.0
        self.scale_c1: float = 0.0
        self.scale_weights_5d: npt.NDArray[np.float64] = np.zeros(5, dtype=np.float64)
        self.mondrian_q_err_table: dict[str, float] = {}
        self._conformal_calibrated: bool = False
        self.stats_loaded: bool = False
        self.stats_path: Path | None = Path(stats_path) if stats_path is not None else None

        self.feature_history: deque[npt.NDArray[np.float64]] = deque(
            maxlen=self.psi_window_size
        )
        self._recent_psi_trade_ids: deque[str] = deque(maxlen=self.psi_window_size)
        self._recent_psi_batch_hashes: deque[bytes] = deque(maxlen=self.psi_window_size)
        self.audit_log: deque[SR262AuditRecord] = deque(maxlen=10_000)
        self._lock = threading.RLock()

        if self.stats_path is not None and self.stats_path.is_file():
            self.load_calibrated_stats(self.stats_path)

    def load_calibrated_stats(self, stats_path: str | Path) -> dict[str, Any]:
        """Loads OAS covariance, 10-decile PSI bins, and split-conformal quantiles safely."""
        p = Path(stats_path)
        raw = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict payload in Guardian stats artifact {p}.")

        def _to_f64(val: Any) -> npt.NDArray[np.float64]:
            if isinstance(val, torch.Tensor):
                return val.detach().cpu().numpy().astype(np.float64, copy=True)
            return np.asarray(val, dtype=np.float64).copy()

        ref_mean = _to_f64(raw["ref_mean_16d"]).reshape(16)
        ref_cov_inv_raw = _to_f64(raw["ref_cov_inv_16d"]).reshape(16, 16)
        if "ref_cov_oas_16d" in raw:
            cov_oas_in = _to_f64(raw["ref_cov_oas_16d"]).reshape(16, 16)
        else:
            cov_oas_in = np.linalg.pinv(ref_cov_inv_raw)
        _, ref_cov_inv, marg_std, cov_inv_d1 = _regularize_cov_oas_correlation_space(cov_oas_in)
        if "ref_marginal_std_16d" in raw:
            marg_std = np.maximum(_to_f64(raw["ref_marginal_std_16d"]).reshape(16), 1e-2)
        bin_edges = _to_f64(raw["psi_bin_edges"]).reshape(16, 11)
        ref_probs = _to_f64(raw["psi_ref_probs"]).reshape(16, 10)

        q_ood = float(raw.get("q_ood_conformal", self.mahalanobis_chi2_threshold))
        q_err = float(raw.get("q_err_conformal", 0.0))
        eps0 = float(raw.get("scale_eps0", 0.0))
        c1 = float(raw.get("scale_c1", 0.0))
        if "scale_weights_5d" in raw:
            w5 = _to_f64(raw["scale_weights_5d"]).reshape(5)
            eps0 = float(w5[0])
            c1 = float(w5[1])
        else:
            w5 = np.array([eps0, c1, 0.0, 0.0, 0.0], dtype=np.float64)
        tau_file = float(raw.get("tau_desk_bps", self.tau_desk_bps))
        alpha_file = float(raw.get("conformal_alpha", self.conformal_alpha))
        raw_mondrian = raw.get("mondrian_q_err_table")
        mondrian_tbl: dict[str, float] = (
            {str(k): float(v) for k, v in raw_mondrian.items()}
            if isinstance(raw_mondrian, dict)
            else {}
        )

        with self._lock:
            self.ref_mean_16d = ref_mean
            self.ref_cov_inv_16d = ref_cov_inv
            self.ref_cov_inv_d1_16d = cov_inv_d1
            self.ref_marginal_std_16d = marg_std
            self.psi_bin_edges = bin_edges
            self.psi_ref_probs = ref_probs
            self.q_ood_conformal = q_ood
            self.q_err_conformal = q_err
            self.scale_eps0 = eps0
            self.scale_c1 = c1
            self.scale_weights_5d = w5.copy()
            self.mondrian_q_err_table = mondrian_tbl
            self.conformal_alpha = alpha_file
            if self.tau_desk_bps == DEFAULT_TAU_DESK_BPS:
                self.tau_desk_bps = tau_file
            self._conformal_calibrated = True
            self.stats_loaded = True
            self.stats_path = p

        return {
            "schema_version": str(raw.get("schema_version", "2.0")),
            "q_ood_conformal": q_ood,
            "q_err_conformal": q_err,
            "scale_eps0": eps0,
            "scale_c1": c1,
            "scale_weights_5d": w5.tolist(),
            "tau_desk_bps": self.tau_desk_bps,
            "n_cal": int(raw.get("n_cal", 0)),
        }

    def fit_reference_distribution(
        self,
        reference_features: npt.NDArray[np.float64] | torch.Tensor,
        *,
        shrinkage: float = 0.10,
    ) -> None:
        """Fits the 16D reference mean, Ledoit-Wolf inverse covariance, and 10-decile PSI bins."""
        if isinstance(reference_features, torch.Tensor):
            x_ref = reference_features.detach().cpu().numpy().astype(np.float64)
        else:
            x_ref = np.asarray(reference_features, dtype=np.float64)
        if x_ref.ndim != 2 or x_ref.shape[1] != 16:
            raise ValueError(f"Expected reference_features of shape (N, 16), got {x_ref.shape}.")

        n_samples = x_ref.shape[0]
        ref_mean = np.mean(x_ref, axis=0)
        diff = x_ref - ref_mean[None, :]
        cov_sample = (diff.T @ diff) / float(max(n_samples - 1, 1))
        diag_target = np.diag(np.maximum(np.diag(cov_sample), 1e-4))
        cov_shrunk = (
            (1.0 - shrinkage) * cov_sample
            + shrinkage * diag_target
            + 1e-4 * np.eye(16, dtype=np.float64)
        )
        _, cov_inv, marg_std, cov_inv_d1 = _regularize_cov_oas_correlation_space(cov_shrunk)

        bin_edges = np.zeros((16, 11), dtype=np.float64)
        ref_probs = np.full((16, 10), 0.10, dtype=np.float64)
        percentiles = np.linspace(0.0, 100.0, 11)
        for j in range(16):
            edges = np.percentile(x_ref[:, j], percentiles).astype(np.float64)
            if len(np.unique(edges[1:-1])) < 9:
                col_std = max(float(np.std(x_ref[:, j])), 1e-3)
                inner = ref_mean[j] + col_std * np.asarray(
                    norm.ppf(np.linspace(0.1, 0.9, 9)), dtype=np.float64
                )
            else:
                inner = edges[1:-1]
            bin_edges[j, 0] = -np.inf
            bin_edges[j, 1:10] = inner
            bin_edges[j, 10] = np.inf
            counts, _ = np.histogram(x_ref[:, j], bins=bin_edges[j])
            probs_j = (counts.astype(np.float64) + 0.5) / (float(max(n_samples, 1)) + 5.0)
            ref_probs[j] = probs_j / np.sum(probs_j)

        with self._lock:
            self.ref_mean_16d = ref_mean
            self.ref_cov_inv_16d = cov_inv
            self.ref_cov_inv_d1_16d = cov_inv_d1
            self.ref_marginal_std_16d = marg_std
            self.psi_bin_edges = bin_edges
            self.psi_ref_probs = ref_probs

    def reset_drift_history(self) -> None:
        """Clears the rolling feature history window and PSI deduplication caches."""
        with self._lock:
            self.feature_history.clear()
            self._recent_psi_trade_ids.clear()
            self._recent_psi_batch_hashes.clear()

    def record_feature_vector(
        self,
        feat_16d: npt.NDArray[np.float64] | Sequence[float],
    ) -> None:
        """Appends a 16D summary feature vector (or batch of vectors) to the rolling PSI window."""
        arr = np.asarray(feat_16d, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != 16:
                raise ValueError(f"Expected 16D feature vector, got shape {arr.shape}.")
            with self._lock:
                self.feature_history.append(arr.copy())
        elif arr.ndim == 2 and arr.shape[1] == 16:
            with self._lock:
                for row in arr:
                    self.feature_history.append(row.copy())
        else:
            raise ValueError(f"Expected shape (16,) or (N, 16), got {arr.shape}.")

    def compute_mahalanobis_d2(
        self,
        feat_16d: npt.NDArray[np.float64],
        *,
        d_assets: int | None = None,
    ) -> float:
        """Computes squared Mahalanobis distance D_M^2(x) = (x - mu)^T Sigma^{-1} (x - mu) in float64.

        When d_assets == 1 (or single-asset summary invariants hold), projects onto the 12D non-degenerate subspace.
        """
        x = np.asarray(feat_16d, dtype=np.float64).reshape(16)
        if not np.all(np.isfinite(x)):
            return 1e6
        is_d1 = (
            (d_assets == 1)
            if d_assets is not None
            else (
                abs(float(x[2]) - float(x[1])) <= 1e-12
                and abs(float(x[5]) - 0.45) <= 1e-9
                and abs(float(x[6]) - 0.25) <= 1e-9
            )
        )
        with self._lock:
            ref_mean = self.ref_mean_16d
            cov_inv = self.ref_cov_inv_d1_16d if is_d1 else self.ref_cov_inv_16d
        diff = x - ref_mean
        d2 = float(diff @ cov_inv @ diff)
        return max(d2, 0.0)

    def effective_psi_threshold(self, n_samples: int | None = None) -> float:
        """Computes Yurdakul-Naranjo finite-sample scaled PSI threshold tau_eff(N) = tau_PSI + (9.0 / N) * (1 + 1.5)."""
        with self._lock:
            n_eff = int(n_samples) if n_samples is not None else len(self.feature_history)
        if n_eff <= 0:
            return float(self.psi_fallback_threshold)
        return float(self.psi_fallback_threshold) + (9.0 / float(n_eff)) * (1.0 + 1.5)

    def compute_psi(
        self,
        observed_batch: npt.NDArray[np.float64] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Computes the 10-decile PSI on `observed_batch` or the rolling `self.feature_history` window."""
        with self._lock:
            bin_edges = self.psi_bin_edges.copy()
            ref_probs = self.psi_ref_probs.copy()
            if observed_batch is not None:
                obs = np.asarray(observed_batch, dtype=np.float64)
            else:
                if len(self.feature_history) < self.psi_min_samples:
                    return 0.0, {name: 0.0 for name in FEATURE_NAMES_16D}
                obs = np.stack(list(self.feature_history), axis=0)
        return compute_psi_10_decile(obs, bin_edges, ref_probs)

    def compute_conformal_half_width_bps(
        self,
        preproc_state: AutocallPreprocState,
        baseline_out: AutocallBaselineOutput | None = None,
    ) -> float:
        """Computes 5D locally-weighted split/Mondrian-conformal error half-width in bps."""
        with self._lock:
            if not self._conformal_calibrated:
                return 0.0
            q_err = self.q_err_conformal
            w5 = self.scale_weights_5d.copy()
            mondrian_tbl = dict(self.mondrian_q_err_table)

        b_out = baseline_out if baseline_out is not None else preproc_state.baseline
        w_np = b_out.proxy_delta_weights.detach().cpu().numpy().astype(np.float64)
        w_abs = np.abs(w_np)
        w_sum = float(np.sum(w_abs))
        if w_sum > 1e-6:
            w_norm = w_abs / w_sum
            h_disp = float(np.clip(1.0 - np.sum(w_norm * w_norm), 0.0, 1.0))
        else:
            h_disp = 0.0

        m_rem = int(preproc_state.m_rem)
        vw_np = (
            b_out.v_w.detach().cpu().numpy().astype(np.float64)
            if m_rem > 0 and b_out.v_w.numel() > 0
            else np.zeros(0, dtype=np.float64)
        )
        v_bar = float(np.mean(vw_np[:m_rem])) if (m_rem > 0 and vw_np.size >= m_rem) else 0.0
        v_last = float(vw_np[m_rem - 1]) if (m_rem > 0 and vw_np.size >= m_rem) else 0.0

        call_code = monitoring_index_to_code(preproc_state.callability_index)
        opt_code = monitoring_index_to_code(
            preproc_state.option_index
            if preproc_state.option_index is not None
            else preproc_state.callability_index
        )
        is_wof = (
            1.0
            if (
                call_code == int(MonitoringIndexCode.WORST_OF)
                or opt_code == int(MonitoringIndexCode.WORST_OF)
            )
            else 0.0
        )
        has_ki_untriggered = (
            1.0
            if (preproc_state.ki_barrier is not None and float(preproc_state.i_ki_past) < 0.5)
            else 0.0
        )
        if mondrian_tbl:
            ki_b = 1 if has_ki_untriggered > 0.5 else 0
            m_b = 0 if m_rem <= 12 else (1 if m_rem <= 25 else 2)
            d_b = 0 if int(preproc_state.d_assets) <= 2 else 1
            q_err = float(mondrian_tbl.get(f"ki{ki_b}_m{m_b}_d{d_b}", q_err))

        phi = np.array(
            [
                1.0,
                math.sqrt(max(h_disp * max(v_bar, 0.0), 0.0)),
                (float(max(m_rem - 1, 0)) / 47.0) * max(v_bar, 0.0),
                is_wof * h_disp,
                has_ki_untriggered * max(float(preproc_state.option_gearing), 0.0) * max(v_last, 0.0),
            ],
            dtype=np.float64,
        )
        scale = float(np.dot(w5, phi))
        return float(max(q_err * scale * 1e4, 0.0))

    @staticmethod
    def _eval_spot_basket_perf(
        m0_np: npt.NDArray[np.float64],
        index_name: str,
        inv_lex_perm: npt.NDArray[np.int64] | None,
    ) -> float:
        """Evaluates monitoring index on current spot performance m0 in original order."""
        idx = str(index_name).strip()
        is_asym = idx in ("RatioAB", "RatioBA", "DiffAB", "DiffBA", "Outperformance", "Spread")
        arr = m0_np[inv_lex_perm] if (is_asym and inv_lex_perm is not None) else m0_np
        res_arr = np.asarray(evaluate_monitoring_index_np(arr[None, :], idx), dtype=np.float64).reshape(-1)
        return float(res_arr[0])

    def compute_min_pin_risk_z(
        self,
        preproc_state: AutocallPreprocState,
        baseline_out: AutocallBaselineOutput | None = None,
    ) -> tuple[float, bool]:
        """Computes imminent log-space barrier pin-risk z-score across short-dated active periods."""
        m_rem = int(preproc_state.m_rem)
        if m_rem == 0 or preproc_state.already_autocalled:
            return 100.0, False

        b_out = baseline_out if baseline_out is not None else preproc_state.baseline
        tau_np = preproc_state.tau_obs.detach().cpu().numpy().astype(np.float64)
        ko_np = preproc_state.ko_barriers.detach().cpu().numpy().astype(np.float64)
        call_np = preproc_state.is_callable.detach().cpu().numpy().astype(np.float64)
        cpn_np = preproc_state.cpn_barriers.detach().cpu().numpy().astype(np.float64)
        guar_np = preproc_state.is_guaranteed_cpn.detach().cpu().numpy().astype(np.float64)
        fw_np = b_out.f_w.detach().cpu().numpy().astype(np.float64)
        vw_np = b_out.v_w.detach().cpu().numpy().astype(np.float64)

        m0_np = preproc_state.moneyness_t0.detach().cpu().numpy().astype(np.float64)
        inv_perm_raw = getattr(preproc_state, "inv_lex_perm", None)
        inv_perm = np.asarray(inv_perm_raw, dtype=np.int64) if inv_perm_raw is not None else None
        w0_call = max(
            self._eval_spot_basket_perf(m0_np, preproc_state.callability_index, inv_perm),
            1e-12,
        )
        eff_cpn_idx = preproc_state.coupon_index or preproc_state.callability_index
        eff_opt_idx = preproc_state.option_index or preproc_state.callability_index
        w0_cpn = max(self._eval_spot_basket_perf(m0_np, eff_cpn_idx, inv_perm), 1e-12)
        w0_opt = max(self._eval_spot_basket_perf(m0_np, eff_opt_idx, inv_perm), 1e-12)

        min_imminent_z = 100.0
        has_imminent = False
        for m in range(m_rem):
            if tau_np[m] > self.pin_risk_tau_max:
                continue
            denom = max(float(vw_np[m]), 1e-12)
            fw_m = max(float(fw_np[m]), 1e-12)

            # 1. Active AutoCall barrier (AND guard prevents false positives on non-callable periods)
            if (call_np[m] > 0.5) and (1e-4 < float(ko_np[m]) < 10.0):
                has_imminent = True
                b_ko = float(ko_np[m])
                z_ko = min(abs(math.log(fw_m / b_ko)), abs(math.log(w0_call / b_ko))) / denom
                if not np.isfinite(z_ko):
                    z_ko = 0.0
                if z_ko < min_imminent_z:
                    min_imminent_z = z_ko

            # 2. Active non-guaranteed Coupon barrier
            if guar_np[m] < 0.5 and 1e-4 < float(cpn_np[m]) < 10.0:
                has_imminent = True
                b_cpn = float(cpn_np[m])
                z_cpn = min(abs(math.log(fw_m / b_cpn)), abs(math.log(w0_cpn / b_cpn))) / denom
                if not np.isfinite(z_cpn):
                    z_cpn = 0.0
                if z_cpn < min_imminent_z:
                    min_imminent_z = z_cpn

            # 3. Active terminal barriers (KnockIn, Put Strike, LowerStrike) on final observation
            if m == m_rem - 1:
                if (
                    preproc_state.ki_barrier is not None
                    and float(preproc_state.ki_barrier) > 1e-4
                    and float(preproc_state.i_ki_past) < 0.5
                ):
                    has_imminent = True
                    b_ki = float(preproc_state.ki_barrier)
                    z_ki = min(abs(math.log(fw_m / b_ki)), abs(math.log(w0_opt / b_ki))) / denom
                    if not np.isfinite(z_ki):
                        z_ki = 0.0
                    if z_ki < min_imminent_z:
                        min_imminent_z = z_ki

                if float(preproc_state.strike) > 1e-4 and float(preproc_state.option_gearing) > 0.0:
                    has_imminent = True
                    k_put = float(preproc_state.strike)
                    z_k = min(abs(math.log(fw_m / k_put)), abs(math.log(w0_opt / k_put))) / denom
                    if not np.isfinite(z_k):
                        z_k = 0.0
                    if z_k < min_imminent_z:
                        min_imminent_z = z_k

                if (
                    preproc_state.lower_strike is not None
                    and float(preproc_state.lower_strike) > 1e-4
                    and float(preproc_state.option_gearing) > 0.0
                ):
                    has_imminent = True
                    k_low = float(preproc_state.lower_strike)
                    z_low = min(abs(math.log(fw_m / k_low)), abs(math.log(w0_opt / k_low))) / denom
                    if not np.isfinite(z_low):
                        z_low = 0.0
                    if z_low < min_imminent_z:
                        min_imminent_z = z_low

        if has_imminent:
            min_z_clamped = float(min(max(min_imminent_z, 0.0), 100.0))
            return min_z_clamped, (min_z_clamped < self.pin_risk_z_min)

        return 100.0, False

    def evaluate_pre_inference(
        self,
        preproc_state: AutocallPreprocState,
        baseline_out: AutocallBaselineOutput | None = None,
        *,
        record_drift: bool | None = None,
    ) -> tuple[
        bool,
        list[AutocallFallbackReason],
        npt.NDArray[np.float64],
        dict[str, Any],
    ]:
        """Executes the Tier 3 Pre-Inference Structural, PSD, Pin-Risk, OOD, Conformal & PSI Gate."""
        b_out = baseline_out if baseline_out is not None else preproc_state.baseline
        reasons: list[AutocallFallbackReason] = []
        warnings: list[str] = []
        for w_s in getattr(getattr(preproc_state, "seasoned_state", None), "warnings", ()):
            if str(w_s) not in warnings:
                warnings.append(str(w_s))
        for _tk, surf_obj in getattr(preproc_state, "vol_surfaces", {}).items():
            for w_v in getattr(surf_obj, "warnings", ()):
                if str(w_v) not in warnings:
                    warnings.append(str(w_v))

        d_assets = int(preproc_state.d_assets)
        m_rem = int(preproc_state.m_rem)

        # 1. Non-finite input tensor check
        input_tensors = [
            preproc_state.spots_t0,
            preproc_state.ref_spots,
            preproc_state.moneyness_t0,
            preproc_state.corr_matrix,
            preproc_state.node_feats,
            preproc_state.edge_feats,
            preproc_state.sched_feats,
            preproc_state.global_feats,
        ]
        has_non_finite_input = any(
            not bool(torch.isfinite(t).all().item()) for t in input_tensors
        )
        if has_non_finite_input:
            reasons.append(AutocallFallbackReason.NON_FINITE_OUTPUT)

        # 2. Basket cardinality & schedule horizon gates
        if d_assets < 1 or d_assets > self.d_max:
            reasons.append(AutocallFallbackReason.BASKET_CARDINALITY_EXCEEDED)

        if m_rem > self.m_max:
            reasons.append(AutocallFallbackReason.SCHEDULE_HORIZON_EXCEEDED)

        if m_rem == 0 or preproc_state.already_autocalled:
            reasons.append(AutocallFallbackReason.ZERO_FUTURE_PERIODS)

        # 3. Exotic monitoring index (incl. mixed-leg indices) & exotic terminal modifier gates
        call_idx_str = str(preproc_state.callability_index).strip()
        cpn_idx_str = (
            str(preproc_state.coupon_index).strip()
            if preproc_state.coupon_index is not None
            else call_idx_str
        )
        opt_idx_str = (
            str(preproc_state.option_index).strip()
            if preproc_state.option_index is not None
            else call_idx_str
        )
        indices_to_check = (call_idx_str, cpn_idx_str, opt_idx_str)
        call_code = monitoring_index_to_code(call_idx_str)
        mixed_index = (
            monitoring_index_to_code(cpn_idx_str) != call_code
            or monitoring_index_to_code(opt_idx_str) != call_code
        )
        if mixed_index or any(idx_name not in SUPPORTED_TIER2_INDICES for idx_name in indices_to_check):
            reasons.append(AutocallFallbackReason.EXOTIC_MONITORING_INDEX)

        has_lower_strike = (
            preproc_state.lower_strike is not None and float(preproc_state.lower_strike) > 0.0
        )
        allow_ac_expiry_active = (
            bool(preproc_state.allow_autocall_on_expiry)
            and m_rem > 0
            and preproc_state.is_callable.numel() > 0
            and float(preproc_state.is_callable[-1].item()) > 0.5
        )
        if (
            float(preproc_state.participation_call_rate) > 0.0
            or float(preproc_state.one_star_coupon) > 0.0
            or (preproc_state.one_star_strike is not None and float(preproc_state.one_star_strike) > 0.0)
            or str(preproc_state.coupon_barrier_type).strip() != "In"
            or str(preproc_state.coupon_barrier_direction).strip() != "Up"
            or has_lower_strike
            or bool(preproc_state.is_long_basket)
            or allow_ac_expiry_active
        ):
            reasons.append(AutocallFallbackReason.EXOTIC_TERMINAL_MODIFIER_ROUTED)

        # 4. Correlation PSD & Dykstra-Higham projection gate
        r_psd, min_eig_raw, min_eig_shrunk, shrunk_applied = project_correlation_to_psd(
            preproc_state.corr_matrix,
            eps_psd=self.eps_psd,
        )
        raw_corr_np = preproc_state.corr_matrix.detach().cpu().numpy().astype(np.float64)
        off_mask = (
            ~np.eye(d_assets, dtype=bool)
            if d_assets >= 2
            else np.zeros((d_assets, d_assets), dtype=bool)
        )
        has_invalid_corr_entry = (
            not bool(np.all(np.isfinite(raw_corr_np)))
            or (bool(np.any(off_mask)) and float(np.max(np.abs(raw_corr_np[off_mask]))) > 1.0 + 1e-6)
        )
        if min_eig_raw < self.severe_psd_threshold or has_invalid_corr_entry:
            reasons.append(AutocallFallbackReason.CORRELATION_NOT_PSD)
            warnings.append(
                f"CORRELATION_SEVERE_NON_PSD: min_eig_raw={min_eig_raw:.6e} < {self.severe_psd_threshold:.6e}; "
                f"projected to R_PSD with min_eig_shrunk={min_eig_shrunk:.6e}."
            )
        elif shrunk_applied:
            warnings.append(
                f"CORRELATION_SHRUNK_TO_PSD: min_eig_raw={min_eig_raw:.6e} -> min_eig_shrunk={min_eig_shrunk:.6e}."
            )

        # 5. Imminent log-space barrier pin-risk gate
        min_pin_z, pin_breached = self.compute_min_pin_risk_z(preproc_state, b_out)
        if pin_breached:
            reasons.append(AutocallFallbackReason.IMMINENT_BARRIER_PIN_RISK)

        # 6. 16D Basket Summary Feature Extraction, Mahalanobis OOD, Active Smile/Domain Box & Conformal Gate
        summary_16d = extract_16d_basket_summary(preproc_state, b_out)
        conformal_hw_bps = self.compute_conformal_half_width_bps(preproc_state, b_out)
        with self._lock:
            eff_ood_thresh = (
                float(self.q_ood_conformal)
                if self._conformal_calibrated
                else float(self.mahalanobis_chi2_threshold)
            )
            conformal_active = bool(self._conformal_calibrated)
            tau_desk_bps = float(self.tau_desk_bps)

        if not bool(np.all(np.isfinite(summary_16d))):
            if AutocallFallbackReason.NON_FINITE_OUTPUT not in reasons:
                reasons.append(AutocallFallbackReason.NON_FINITE_OUTPUT)
            mahalanobis_d2 = 1e6
        else:
            mahalanobis_d2 = self.compute_mahalanobis_d2(summary_16d, d_assets=d_assets)
            max_vol_val = float(summary_16d[AutocallSummaryCol.MAX_ATM_VOL])
            max_div_lev = float(summary_16d[AutocallSummaryCol.MAX_CASH_DIV_LEVERAGE])
            min_log_m = float(summary_16d[AutocallSummaryCol.MIN_LOG_MONEYNESS])
            max_log_m = float(summary_16d[AutocallSummaryCol.MAX_LOG_MONEYNESS])

            domain_box_ood = (
                max_vol_val > 1.00
                or max_div_lev > math.log(1.0 / (1.0 - 0.35))
                or min_log_m < math.log(0.15)
                or max_log_m > math.log(2.50)
                or float(preproc_state.option_gearing) > 2.0
            )

            # Check active-asset smile box bounds (scoped to |w_proxy_i| > 1e-3 & K in [0.4 F_i, 2.0 F_i])
            if not domain_box_ood and d_assets > 0:
                w_proxy_np = b_out.proxy_delta_weights.detach().cpu().numpy().astype(np.float64)
                node_np = preproc_state.node_feats[0, :d_assets, :].detach().cpu().numpy().astype(np.float64)
                if float(np.max(np.abs(node_np[:, int(AutocallNodeCol.REPO_SPREAD_AVG)]))) > 0.15:
                    domain_box_ood = True

                fwd_np = (
                    preproc_state.dividend_schedule.forward_price.detach().cpu().numpy().astype(np.float64)
                    if preproc_state.dividend_schedule.forward_price.numel() > 0
                    else None
                )
                for i in range(d_assets):
                    if abs(float(w_proxy_np[i])) <= 1e-3:
                        continue
                    skew_rho = abs(float(node_np[i, int(AutocallNodeCol.SSVI_SKEW_RHO)]))
                    curv_eta = float(node_np[i, int(AutocallNodeCol.SSVI_CURV_ETA)])
                    pc1 = abs(float(node_np[i, int(AutocallNodeCol.SMILE_RESIDUAL_PC1)]))
                    pc2 = abs(float(node_np[i, int(AutocallNodeCol.SMILE_RESIDUAL_PC2)]))
                    if skew_rho > 0.75 or curv_eta > 3.5 or pc1 > 0.25 or pc2 > 0.25:
                        domain_box_ood = True
                        break
                    ticker_i = preproc_state.sorted_identifiers[i]
                    surf_i = preproc_state.vol_surfaces.get(ticker_i)
                    if (
                        surf_i is not None
                        and fwd_np is not None
                        and fwd_np.shape[0] > i
                        and fwd_np.shape[1] > 0
                    ):
                        f_i = float(fwd_np[i, -1])
                        k_mask = (surf_i.strikes >= 0.4 * f_i) & (surf_i.strikes <= 2.0 * f_i)
                        if np.any(k_mask):
                            q_act = surf_i.quotes[:, k_mask]
                            q_ratio = float(np.max(q_act)) / max(float(np.min(q_act)), 1e-12)
                            if q_ratio > 5.0:
                                domain_box_ood = True
                                break

            # Check zero-rate bounds and quanto drift bounds
            if not domain_box_ood and m_rem > 0:
                df_pay_np = preproc_state.df_pay.detach().cpu().numpy().astype(np.float64)
                tau_pay_np = preproc_state.tau_pay.detach().cpu().numpy().astype(np.float64)
                for m_idx in range(m_rem):
                    t_p = float(tau_pay_np[m_idx])
                    if t_p >= 0.02:
                        r_zero_m = -math.log(max(float(df_pay_np[m_idx]), 1e-12)) / t_p
                        if r_zero_m < -0.01 or r_zero_m > 0.30:
                            domain_box_ood = True
                            break

            if not domain_box_ood and preproc_state.quanto_meta:
                for q_ticker, q_meta in preproc_state.quanto_meta.items():
                    if q_ticker not in preproc_state.sorted_identifiers:
                        continue
                    idx_q = preproc_state.sorted_identifiers.index(q_ticker)
                    rho_fx = abs(float(q_meta.get("rho_s_fx", 0.0)))
                    sig_fx_vals = q_meta.get("sig_fx_obs", ())
                    sig_fx_max = max((abs(float(v)) for v in sig_fx_vals), default=0.0)
                    sig_eq_max = (
                        float(torch.max(preproc_state.iv_at_fwd[idx_q]).item())
                        if preproc_state.iv_at_fwd.numel() > 0
                        else 0.0
                    )
                    if rho_fx * sig_fx_max * sig_eq_max > 0.10:
                        domain_box_ood = True
                        break

            if domain_box_ood:
                mahalanobis_d2 = max(mahalanobis_d2, eff_ood_thresh + 10.0)

        if mahalanobis_d2 > eff_ood_thresh:
            reasons.append(AutocallFallbackReason.MAHALANOBIS_OOD)

        if conformal_active and conformal_hw_bps > tau_desk_bps:
            reasons.append(AutocallFallbackReason.CONFORMAL_UNCERTAINTY_BREACH)
            warnings.append(
                f"CONFORMAL_UNCERTAINTY_BREACH: conformal_half_width_bps={conformal_hw_bps:.2f} > {tau_desk_bps:.2f}."
            )

        do_record = (
            self.record_drift_by_default if record_drift is None else bool(record_drift)
        )
        if do_record and bool(np.all(np.isfinite(summary_16d))):
            t_id = str(preproc_state.trade_info) if preproc_state.trade_info is not None else ""
            should_push = True
            if t_id:
                feat_sig = hashlib.blake2b(np.round(summary_16d, 6).tobytes(), digest_size=8).hexdigest()
                dedup_id = f"{t_id}:{feat_sig}"
                with self._lock:
                    if dedup_id in self._recent_psi_trade_ids or t_id in self._recent_psi_trade_ids:
                        should_push = False
                    else:
                        self._recent_psi_trade_ids.append(dedup_id)
            if should_push:
                self.record_feature_vector(summary_16d)

        max_psi, psi_per_feature = self.compute_psi()
        with self._lock:
            n_hist = len(self.feature_history)
        eff_psi_thresh = (
            self.effective_psi_threshold(n_hist)
            if n_hist > 0
            else float(self.psi_fallback_threshold)
        )
        if max_psi >= eff_psi_thresh:
            if self.psi_hard_circuit_breaker:
                reasons.append(AutocallFallbackReason.PSI_DRIFT_BREACH)
            warnings.append(
                f"PSI_DRIFT_BREACH: max_psi={max_psi:.4f} >= {eff_psi_thresh:.4f}."
            )
        elif max_psi >= self.psi_warning_threshold:
            warnings.append(
                f"PSI_MODERATE_DRIFT_WARNING: max_psi={max_psi:.4f} >= {self.psi_warning_threshold:.4f}."
            )

        passed = len(reasons) == 0
        diagnostics: dict[str, Any] = {
            "min_corr_eigenvalue_raw": float(min_eig_raw),
            "min_corr_eigenvalue_shrunk": float(min_eig_shrunk),
            "correlation_shrunk_applied": bool(shrunk_applied),
            "mahalanobis_d2": float(mahalanobis_d2),
            "max_psi_score": float(max_psi),
            "psi_per_feature": psi_per_feature,
            "min_pin_risk_z": float(min_pin_z),
            "conformal_half_width_bps": float(conformal_hw_bps),
            "summary_features_16d": summary_16d,
            "warnings": warnings,
        }
        return passed, reasons, r_psd, diagnostics

    def evaluate_post_inference(
        self,
        surrogate_out: AutocallSurrogateOutput | dict[str, Any] | None,
        postproc_out: AutocallPostprocOutput | dict[str, Any] | None,
        preproc_state: AutocallPreprocState,
        baseline_out: AutocallBaselineOutput | None = None,
    ) -> tuple[bool, list[AutocallFallbackReason]]:
        """Executes the Tier 3 Post-Inference Probability Simplex, Put-Spread Bound & Funded Residual Gate."""
        reasons: list[AutocallFallbackReason] = []
        m_rem = int(preproc_state.m_rem)
        b_out = baseline_out if baseline_out is not None else preproc_state.baseline

        p_arr: npt.NDArray[np.float64] | None = None
        q_arr: npt.NDArray[np.float64] | None = None
        l_put_val: float | None = None
        pv_surrogate: float | None = None

        if isinstance(surrogate_out, AutocallSurrogateOutput):
            p_arr = (
                surrogate_out.stopping_probs.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1)[:m_rem]
            )
            q_arr = (
                surrogate_out.coupon_probs.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1)[:m_rem]
            )
            l_put_val = float(
                surrogate_out.put_loss.detach().cpu().numpy().astype(np.float64).reshape(-1)[0]
            )
        elif isinstance(surrogate_out, dict):
            if "stopping_probs" in surrogate_out or "p_stop" in surrogate_out:
                raw_p = surrogate_out.get("stopping_probs", surrogate_out.get("p_stop"))
                p_arr = np.asarray(
                    raw_p.detach().cpu().numpy() if isinstance(raw_p, torch.Tensor) else raw_p,
                    dtype=np.float64,
                ).reshape(-1)[:m_rem]
            if "coupon_probs" in surrogate_out or "q_cpn" in surrogate_out:
                raw_q = surrogate_out.get("coupon_probs", surrogate_out.get("q_cpn"))
                q_arr = np.asarray(
                    raw_q.detach().cpu().numpy() if isinstance(raw_q, torch.Tensor) else raw_q,
                    dtype=np.float64,
                ).reshape(-1)[:m_rem]
            if "put_loss" in surrogate_out or "l_put" in surrogate_out:
                raw_l = surrogate_out.get("put_loss", surrogate_out.get("l_put"))
                l_put_val = float(
                    np.asarray(
                        raw_l.detach().cpu().numpy() if isinstance(raw_l, torch.Tensor) else raw_l,
                        dtype=np.float64,
                    ).reshape(-1)[0]
                )

        if isinstance(postproc_out, AutocallPostprocOutput):
            pv_surrogate = float(postproc_out.product_value.detach().cpu().item())
            if p_arr is None:
                p_arr = (
                    postproc_out.p_stop.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(-1)[:m_rem]
                )
            if q_arr is None:
                q_arr = (
                    postproc_out.q_cpn.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                    .reshape(-1)[:m_rem]
                )
            if l_put_val is None:
                l_put_val = float(postproc_out.l_put.detach().cpu().item())
        elif isinstance(postproc_out, dict):
            if "product_value" in postproc_out or "ProductValue" in postproc_out:
                raw_pv = postproc_out.get("product_value", postproc_out.get("ProductValue"))
                if raw_pv is not None:
                    pv_surrogate = float(
                        raw_pv.detach().cpu().item()
                        if isinstance(raw_pv, torch.Tensor)
                        else float(raw_pv)
                    )
            if p_arr is None and ("p_stop" in postproc_out or "stopping_probs" in postproc_out):
                raw_p = postproc_out.get("p_stop", postproc_out.get("stopping_probs"))
                p_arr = np.asarray(
                    raw_p.detach().cpu().numpy() if isinstance(raw_p, torch.Tensor) else raw_p,
                    dtype=np.float64,
                ).reshape(-1)[:m_rem]
            if q_arr is None and ("q_cpn" in postproc_out or "coupon_probs" in postproc_out):
                raw_q = postproc_out.get("q_cpn", postproc_out.get("coupon_probs"))
                q_arr = np.asarray(
                    raw_q.detach().cpu().numpy() if isinstance(raw_q, torch.Tensor) else raw_q,
                    dtype=np.float64,
                ).reshape(-1)[:m_rem]
            if l_put_val is None and ("l_put" in postproc_out or "put_loss" in postproc_out):
                raw_l = postproc_out.get("l_put", postproc_out.get("put_loss"))
                l_put_val = float(
                    np.asarray(
                        raw_l.detach().cpu().numpy() if isinstance(raw_l, torch.Tensor) else raw_l,
                        dtype=np.float64,
                    ).reshape(-1)[0]
                )

        if p_arr is None:
            p_arr = np.zeros(m_rem, dtype=np.float64)
        if q_arr is None:
            q_arr = np.zeros(m_rem, dtype=np.float64)
        if l_put_val is None:
            l_put_val = 0.0

        if pv_surrogate is None:
            if np.all(np.isfinite(p_arr)) and np.all(np.isfinite(q_arr)) and np.isfinite(l_put_val):
                asm = assemble_autocall_cashflows_and_pv(
                    preproc_state,
                    p_stop=torch.as_tensor(p_arr, dtype=torch.float64),
                    q_cpn=torch.as_tensor(q_arr, dtype=torch.float64),
                    l_put=torch.tensor(l_put_val, dtype=torch.float64),
                    build_cashflow_items=False,
                )
                pv_surrogate = float(asm.product_value.detach().cpu().item())
            else:
                pv_surrogate = float("nan")

        # 1. Non-finite check
        if (
            not bool(np.all(np.isfinite(p_arr)))
            or not bool(np.all(np.isfinite(q_arr)))
            or not np.isfinite(l_put_val)
            or not np.isfinite(pv_surrogate)
        ):
            reasons.append(AutocallFallbackReason.NON_FINITE_OUTPUT)
            return False, reasons

        # 2. Simplex & probability bounds check
        sum_p = float(np.sum(p_arr)) if m_rem > 0 else 0.0
        p_out_of_bounds = m_rem > 0 and bool(
            np.any(p_arr < -1e-7) or np.any(p_arr > 1.0 + 1e-7)
        )
        q_out_of_bounds = m_rem > 0 and bool(
            np.any(q_arr < -1e-7) or np.any(q_arr > 1.0 + 1e-7)
        )
        if sum_p > 1.0 + 1e-6 or p_out_of_bounds or q_out_of_bounds:
            reasons.append(AutocallFallbackReason.POST_INFERENCE_SIMPLEX_VIOLATION)

        # 3. Terminal put-spread bound check: 0 <= L_put <= max(K_put - K_low, 0) * S_{M_rem}
        surv_term = max(1.0 - sum_p, 0.0)
        if isinstance(surrogate_out, AutocallSurrogateOutput) and m_rem > 0:
            surv_from_out = float(
                surrogate_out.survival_probs.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1)[min(m_rem - 1, surrogate_out.survival_probs.shape[-1] - 1)]
            )
            if np.isfinite(surv_from_out):
                surv_term = max(min(surv_term, surv_from_out), 0.0)

        k_put = max(float(preproc_state.strike), 0.0)
        k_low = (
            max(float(preproc_state.lower_strike), 0.0)
            if preproc_state.lower_strike is not None
            else 0.0
        )
        k_eff = max(k_put - k_low, 0.0)
        max_put_bound = k_eff * surv_term + 1e-6
        if l_put_val < -1e-7 or l_put_val > max_put_bound:
            reasons.append(AutocallFallbackReason.POST_INFERENCE_PUT_BOUND_VIOLATION)

        # 4. Funded-vs-funded proxy residual outlier check (EXT-012)
        pv_proxy_unfunded = float(b_out.product_value_proxy.detach().cpu().item())
        if preproc_state.funding_leg is not None:
            asm_proxy = assemble_autocall_cashflows_and_pv(
                preproc_state,
                build_cashflow_items=False,
            )
            pv_proxy = float(asm_proxy.product_value.detach().cpu().item())
        else:
            pv_proxy = pv_proxy_unfunded

        if self.residual_threshold is None:
            w_np = b_out.proxy_delta_weights.detach().cpu().numpy().astype(np.float64)
            w_abs = np.abs(w_np)
            w_sum = max(float(np.sum(w_abs)), 1e-6)
            w_norm = w_abs / w_sum
            h_disp = float(np.clip(1.0 - np.sum(w_norm * w_norm), 0.0, 1.0))
            v_bar = (
                float(np.mean(b_out.v_w.detach().cpu().numpy().astype(np.float64)))
                if m_rem > 0
                else 0.0
            )
            tau_res = min(0.0050 + 0.0350 * h_disp * math.sqrt(max(v_bar, 1e-4)), 0.0200)
        else:
            tau_res = float(self.residual_threshold)
        if abs(pv_surrogate - pv_proxy) > tau_res:
            reasons.append(AutocallFallbackReason.POST_INFERENCE_RESIDUAL_OUTLIER)

        return len(reasons) == 0, reasons

    @staticmethod
    def _apply_shrunk_correlation_to_state(
        preproc_state: AutocallPreprocState,
        r_psd: npt.NDArray[np.float64],
    ) -> AutocallBaselineOutput:
        """Updates `preproc_state` in-place with the PSD-projected correlation matrix `r_psd` in float64."""
        d_assets = int(preproc_state.d_assets)
        device = preproc_state.spots_t0.device
        corr_t = torch.as_tensor(r_psd, dtype=torch.float64, device=device)
        preproc_state.corr_matrix = corr_t

        if d_assets > 0:
            edge_clone = preproc_state.edge_feats.clone()
            edge_clone[0, :d_assets, :d_assets, int(AutocallEdgeCol.CORR_RHO)] = corr_t
            if preproc_state.m_rem > 0:
                v_term = preproc_state.v_perf[:, -1]
                cov_t = corr_t * v_term.unsqueeze(1) * v_term.unsqueeze(0)
                edge_clone[0, :d_assets, :d_assets, int(AutocallEdgeCol.COVARIANCE_TERM)] = cov_t
            preproc_state.edge_feats = edge_clone

        new_baseline = compute_analytical_autocall_baseline(preproc_state)
        preproc_state.baseline = new_baseline
        if d_assets > 0:
            node_clone = preproc_state.node_feats.clone()
            node_clone[0, :d_assets, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)] = (
                new_baseline.proxy_delta_weights.to(dtype=node_clone.dtype, device=device)
            )
            preproc_state.node_feats = node_clone
        m_rem = int(preproc_state.m_rem)
        if m_rem > 0:
            sched_clone = preproc_state.sched_feats.clone()
            sched_clone[0, :m_rem, int(AutocallSchedCol.STULZ_FWD_W)] = torch.log(
                new_baseline.f_w.clamp(min=1e-12)
            ).to(dtype=sched_clone.dtype, device=device)
            sched_clone[0, :m_rem, int(AutocallSchedCol.STULZ_VOL_W)] = new_baseline.v_w.to(
                dtype=sched_clone.dtype, device=device
            )
            sched_clone[0, :m_rem, int(AutocallSchedCol.PROXY_KO_PROB)] = new_baseline.p_proxy.to(
                dtype=sched_clone.dtype, device=device
            )
            sched_clone[0, :m_rem, int(AutocallSchedCol.PROXY_CPN_PROB)] = new_baseline.q_proxy.to(
                dtype=sched_clone.dtype, device=device
            )
            preproc_state.sched_feats = sched_clone
        glob_clone = preproc_state.global_feats.clone()
        glob_clone[0, int(AutocallGlobalCol.L_PUT_PROXY)] = new_baseline.l_put_proxy.to(
            dtype=glob_clone.dtype, device=device
        )
        preproc_state.global_feats = glob_clone
        return new_baseline

    def _extract_batch_16d_summary(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> npt.NDArray[np.float64]:
        """Vectorized 16D summary feature extraction for tensor/Arrow batches."""
        node_np = node_feats.detach().cpu().numpy().astype(np.float64)
        edge_np = edge_feats.detach().cpu().numpy().astype(np.float64)
        amask_np = asset_mask.detach().cpu().numpy().astype(np.float64) > 0.5
        sched_np = sched_feats.detach().cpu().numpy().astype(np.float64)
        smask_np = sched_mask.detach().cpu().numpy().astype(np.float64) > 0.5
        glob_np = global_feats.detach().cpu().numpy().astype(np.float64)

        b_sz = node_np.shape[0]
        out = np.zeros((b_sz, 16), dtype=np.float64)
        a_cnt = np.maximum(np.sum(amask_np, axis=1), 1).astype(np.float64)
        log_m = node_np[:, :, int(AutocallNodeCol.LOG_MONEYNESS)]
        log_m_masked_min = np.where(amask_np, log_m, np.inf)
        log_m_masked_max = np.where(amask_np, log_m, -np.inf)
        has_a = np.any(amask_np, axis=1)
        out[:, 0] = np.where(has_a, np.sum(np.where(amask_np, log_m, 0.0), axis=1) / a_cnt, 0.0)
        out[:, 1] = np.where(has_a, np.min(log_m_masked_min, axis=1), 0.0)
        out[:, 2] = np.where(has_a, np.max(log_m_masked_max, axis=1), 0.0)

        iv_s = node_np[:, :, int(AutocallNodeCol.IV_SHORT_ATM)]
        iv_t = node_np[:, :, int(AutocallNodeCol.IV_TERM_ATM)]
        out[:, 3] = np.where(has_a, np.sum(np.where(amask_np, iv_t, 0.0), axis=1) / a_cnt, 0.0)
        out[:, 4] = np.where(
            has_a,
            np.max(np.where(amask_np, np.maximum(iv_s, iv_t), -np.inf), axis=1),
            0.0,
        )
        div_l = node_np[:, :, int(AutocallNodeCol.CASH_DIV_LEVERAGE)]
        out[:, 7] = np.where(has_a, np.max(np.where(amask_np, div_l, -np.inf), axis=1), 0.0)

        d_max = node_np.shape[1]
        eye_d = np.eye(d_max, dtype=bool)[None, :, :]
        pair_mask = amask_np[:, :, None] & amask_np[:, None, :] & (~eye_d)
        pair_cnt = np.sum(pair_mask, axis=(1, 2))
        has_pair = pair_cnt > 0
        corr_mat = edge_np[:, :, :, int(AutocallEdgeCol.CORR_RHO)]
        out[:, 5] = np.where(
            has_pair,
            np.sum(np.where(pair_mask, corr_mat, 0.0), axis=(1, 2)) / np.maximum(pair_cnt, 1),
            0.45,
        )
        out[:, 6] = np.where(
            has_pair,
            np.min(np.where(pair_mask, corr_mat, np.inf), axis=(1, 2)),
            0.25,
        )

        m_cnt = np.sum(smask_np, axis=1)
        has_m = m_cnt > 0
        last_m_idx = np.clip(m_cnt - 1, 0, max(sched_np.shape[1] - 1, 0))
        rows_idx = np.arange(b_sz)
        if sched_np.shape[1] > 0:
            tau_last = sched_np[rows_idx, last_m_idx, int(AutocallSchedCol.TAU_OBS)]
            fw_last = sched_np[rows_idx, last_m_idx, int(AutocallSchedCol.STULZ_FWD_W)]
            vw_last = sched_np[rows_idx, last_m_idx, int(AutocallSchedCol.STULZ_VOL_W)]
        else:
            tau_last = np.zeros(b_sz, dtype=np.float64)
            fw_last = np.zeros(b_sz, dtype=np.float64)
            vw_last = np.full(b_sz, 0.25, dtype=np.float64)
        out[:, 8] = np.where(has_m, tau_last, 0.0)

        log_ko = sched_np[:, :, int(AutocallSchedCol.LOG_KO_BARRIER)]
        ko_non_finite = smask_np & (~(log_ko >= -50.0) | ~(log_ko <= 50.0))
        is_call = smask_np & (
            ((sched_np[:, :, int(AutocallSchedCol.IS_CALLABLE_FLAG)] > 0.5) & ~(log_ko >= math.log(100.0)))
            | ko_non_finite
        )
        call_cnt = np.sum(is_call, axis=1)
        out[:, 9] = np.where(
            call_cnt > 0,
            np.sum(np.where(is_call, log_ko, 0.0), axis=1) / np.maximum(call_cnt, 1),
            0.05,
        )
        out[:, 9] = np.where(np.any(ko_non_finite, axis=1), np.nan, out[:, 9])

        log_cpn = sched_np[:, :, int(AutocallSchedCol.LOG_CPN_BARRIER)]
        cpn_non_finite = smask_np & (~(log_cpn >= -50.0) | ~(log_cpn <= 50.0))
        guar = sched_np[:, :, int(AutocallSchedCol.IS_GUARANTEED_CPN)]
        cpn_act = smask_np & (
            ((guar < 0.5) & ~(log_cpn <= math.log(1e-4) + 1e-4))
            | cpn_non_finite
        )
        cpn_cnt = np.sum(cpn_act, axis=1)
        out[:, 10] = np.where(
            cpn_cnt > 0,
            np.sum(np.where(cpn_act, log_cpn, 0.0), axis=1) / np.maximum(cpn_cnt, 1),
            -0.50,
        )
        out[:, 10] = np.where(np.any(cpn_non_finite, axis=1), np.nan, out[:, 10])
        out[:, 11] = glob_np[:, int(AutocallGlobalCol.K_PUT)]
        out[:, 12] = np.where(has_m, fw_last, out[:, 1])
        out[:, 13] = np.where(has_m, vw_last, 0.25)
        out[:, 14] = glob_np[:, int(AutocallGlobalCol.C_MEM_0)]
        out[:, 15] = np.abs(glob_np[:, int(AutocallGlobalCol.ABS_KO_SPREAD)])
        return out

    def evaluate_tensor_batch_pre(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        asset_mask: torch.Tensor,
        sched_feats: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
        *,
        record_drift: bool | None = None,
    ) -> dict[str, Any]:
        """Vectorized SR 26-2 pre-inference batch screening for Arrow/tensor batch endpoints."""
        node_np = node_feats.detach().cpu().numpy().astype(np.float64, copy=False)
        edge_np = edge_feats.detach().cpu().numpy().astype(np.float64, copy=False)
        amask_np = asset_mask.detach().cpu().numpy().astype(np.float64, copy=False) > 0.5
        sched_np = sched_feats.detach().cpu().numpy().astype(np.float64, copy=False)
        smask_np = sched_mask.detach().cpu().numpy().astype(np.float64, copy=False) > 0.5
        glob_np = global_feats.detach().cpu().numpy().astype(np.float64, copy=False)

        b_sz = node_np.shape[0]
        summary_batch = self._extract_batch_16d_summary(
            node_feats,
            edge_feats,
            asset_mask,
            sched_feats,
            sched_mask,
            global_feats,
        )
        vw_all = sched_np[:, :, int(AutocallSchedCol.STULZ_VOL_W)]

        node_active_ok = np.all(
            np.where(amask_np[:, :, None], np.isfinite(node_np), True),
            axis=(1, 2),
        )
        pair_active = amask_np[:, :, None] & amask_np[:, None, :]
        edge_active_ok = np.all(
            np.where(pair_active[:, :, :, None], np.isfinite(edge_np), True),
            axis=(1, 2, 3),
        )
        sched_active_ok = np.all(
            np.where(smask_np[:, :, None], np.isfinite(sched_np), True),
            axis=(1, 2),
        )
        glob_active_ok = np.all(np.isfinite(glob_np), axis=1)
        finite_mask = (
            np.all(np.isfinite(summary_batch), axis=1)
            & node_active_ok
            & edge_active_ok
            & sched_active_ok
            & glob_active_ok
        )
        with self._lock:
            ref_mean = self.ref_mean_16d
            cov_inv = self.ref_cov_inv_16d
            cov_inv_d1 = self.ref_cov_inv_d1_16d
            eff_ood = (
                float(self.q_ood_conformal)
                if self._conformal_calibrated
                else float(self.mahalanobis_chi2_threshold)
            )
            conformal_active = bool(self._conformal_calibrated)
            q_err = float(self.q_err_conformal)
            w5 = self.scale_weights_5d.copy()
            mondrian_tbl = dict(self.mondrian_q_err_table)
            tau_desk = float(self.tau_desk_bps)

        diff = summary_batch - ref_mean[None, :]
        d_cnt_vec = np.sum(amask_np, axis=1)
        is_d1_vec = d_cnt_vec == 1
        d2_full = np.maximum(np.sum((diff @ cov_inv) * diff, axis=1), 0.0)
        if bool(np.any(is_d1_vec)):
            d2_d1 = np.maximum(np.sum((diff @ cov_inv_d1) * diff, axis=1), 0.0)
            d2_vec = np.where(is_d1_vec, d2_d1, d2_full)
        else:
            d2_vec = d2_full
        d2_vec = np.where(finite_mask, d2_vec, 1e6)

        if conformal_active:
            w_np = node_np[:, :, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)]
            w_abs = np.where(amask_np, np.abs(w_np), 0.0)
            w_sum = np.sum(w_abs, axis=1, keepdims=True)
            has_w = w_sum.squeeze(1) > 1e-6
            w_norm = w_abs / np.maximum(w_sum, 1e-6)
            h_disp = np.where(has_w, np.clip(1.0 - np.sum(w_norm * w_norm, axis=1), 0.0, 1.0), 0.0)
            raw_m_cnt = np.sum(smask_np, axis=1)
            m_cnt = np.maximum(raw_m_cnt, 1)
            v_bar = np.sum(np.where(smask_np, vw_all, 0.0), axis=1) / m_cnt
            if glob_np.shape[1] > int(AutocallGlobalCol.OPT_INDEX_ID):
                call_id = np.round(glob_np[:, int(AutocallGlobalCol.CALL_INDEX_ID)])
                opt_id = np.round(glob_np[:, int(AutocallGlobalCol.OPT_INDEX_ID)])
            else:
                call_id = np.zeros(b_sz, dtype=np.float64)
                opt_id = np.zeros(b_sz, dtype=np.float64)
            is_wof = (
                (call_id == int(MonitoringIndexCode.WORST_OF))
                | (opt_id == int(MonitoringIndexCode.WORST_OF))
            ).astype(np.float64)
            has_ki_b = glob_np[:, int(AutocallGlobalCol.HAS_KI_FLAG)] > 0.5
            i_kip_b = glob_np[:, int(AutocallGlobalCol.I_KI_PAST)] < 0.5
            g_put_b = np.maximum(glob_np[:, int(AutocallGlobalCol.G_PUT)], 0.0)
            v_last_b = summary_batch[:, 13] * np.where(raw_m_cnt > 0, 1.0, 0.0)
            phi_mat = np.stack(
                [
                    np.ones(b_sz, dtype=np.float64),
                    np.sqrt(np.maximum(h_disp * np.maximum(v_bar, 0.0), 0.0)),
                    (np.maximum(raw_m_cnt.astype(np.float64) - 1.0, 0.0) / 47.0)
                    * np.maximum(v_bar, 0.0),
                    is_wof * h_disp,
                    np.where(has_ki_b & i_kip_b, 1.0, 0.0)
                    * g_put_b
                    * np.maximum(v_last_b, 0.0),
                ],
                axis=1,
            )
            if mondrian_tbl:
                ki_untrig_vec = has_ki_b & i_kip_b
                q_err_vec = np.array(
                    [
                        float(
                            mondrian_tbl.get(
                                f"ki{1 if bool(ki_u) else 0}_m{0 if int(m_c) <= 12 else (1 if int(m_c) <= 25 else 2)}_d{0 if int(d_c) <= 2 else 1}",
                                q_err,
                            )
                        )
                        for ki_u, m_c, d_c in zip(ki_untrig_vec, raw_m_cnt, d_cnt_vec)
                    ],
                    dtype=np.float64,
                )
            else:
                q_err_vec = np.full(b_sz, q_err, dtype=np.float64)
            conf_hw_vec = np.maximum(q_err_vec * (phi_mat @ w5) * 1e4, 0.0)
        else:
            conf_hw_vec = np.zeros(b_sz, dtype=np.float64)

        do_record = self.record_drift_by_default if record_drift is None else bool(record_drift)
        if do_record and b_sz > 0:
            finite_rows = summary_batch[finite_mask]
            if len(finite_rows) > 0:
                if len(finite_rows) > 1 and bool(np.all(finite_rows == finite_rows[0])):
                    uniq_vals = np.round(finite_rows[:1], 6)
                    distinct_rows = finite_rows[:1]
                else:
                    rounded = np.round(finite_rows, 6)
                    uniq_vals, uniq_idx = np.unique(rounded, axis=0, return_index=True)
                    distinct_rows = finite_rows[np.sort(uniq_idx)]
                batch_sig = hashlib.blake2b(uniq_vals.tobytes(), digest_size=16).digest()
                should_record_batch = True
                with self._lock:
                    if batch_sig in self._recent_psi_batch_hashes:
                        should_record_batch = False
                    else:
                        self._recent_psi_batch_hashes.append(batch_sig)
                if should_record_batch:
                    stride = max(
                        1, len(distinct_rows) // min(len(distinct_rows), self.psi_window_size)
                    )
                    self.record_feature_vector(distinct_rows[::stride])

        max_psi, _ = self.compute_psi()
        with self._lock:
            n_hist = len(self.feature_history)
        eff_psi_thresh = (
            self.effective_psi_threshold(n_hist)
            if n_hist > 0
            else float(self.psi_fallback_threshold)
        )
        ood_flags = d2_vec > eff_ood
        conf_flags = conf_hw_vec > tau_desk if conformal_active else np.zeros(b_sz, dtype=bool)
        return {
            "batch_size": b_sz,
            "max_mahalanobis_d2": float(np.max(d2_vec)) if b_sz > 0 else 0.0,
            "mean_mahalanobis_d2": float(np.mean(d2_vec)) if b_sz > 0 else 0.0,
            "ood_count": int(np.sum(ood_flags)),
            "max_conformal_half_width_bps": float(np.max(conf_hw_vec)) if b_sz > 0 else 0.0,
            "conformal_breach_count": int(np.sum(conf_flags)),
            "max_psi_score": float(max_psi),
            "psi_breached": bool(max_psi >= eff_psi_thresh),
            "non_finite_count": int(np.sum(~finite_mask)),
        }

    def evaluate_tensor_batch_post(
        self,
        stopping_probs: torch.Tensor,
        coupon_probs: torch.Tensor,
        put_loss: torch.Tensor,
        product_value: torch.Tensor,
        sched_mask: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> dict[str, Any]:
        """Vectorized SR 26-2 post-inference simplex, put-spread & finiteness audit for batches."""
        with torch.no_grad():
            finite_ok = (
                torch.isfinite(stopping_probs).all(dim=1)
                & torch.isfinite(coupon_probs).all(dim=1)
                & torch.isfinite(put_loss)
                & torch.isfinite(product_value)
            )
            sm_b = sched_mask > 0.5
            p_act = torch.where(sm_b, stopping_probs, torch.zeros_like(stopping_probs))
            q_act = torch.where(sm_b, coupon_probs, torch.zeros_like(coupon_probs))
            sum_p = p_act.sum(dim=1)
            simplex_ok = (
                (sum_p <= 1.0 + 1e-6)
                & ((p_act >= -1e-7) & (p_act <= 1.0 + 1e-7)).all(dim=1)
                & ((q_act >= -1e-7) & (q_act <= 1.0 + 1e-7)).all(dim=1)
            )
            k_put = global_feats[:, int(AutocallGlobalCol.K_PUT)].clamp(min=0.0)
            if global_feats.shape[1] > int(AutocallGlobalCol.LOWER_STRIKE):
                k_low = global_feats[:, int(AutocallGlobalCol.LOWER_STRIKE)].clamp(min=0.0)
            else:
                k_low = torch.zeros_like(k_put)
            k_eff = (k_put - k_low).clamp(min=0.0)
            surv_term = (1.0 - sum_p).clamp(min=0.0)
            put_ok = (put_loss >= -1e-7) & (put_loss <= k_eff * surv_term + 1e-6)
            counts = torch.stack(
                [(~finite_ok).sum(), (~simplex_ok).sum(), (~put_ok).sum()]
            ).tolist()
        nf_cnt, sim_cnt, put_cnt = int(counts[0]), int(counts[1]), int(counts[2])
        return {
            "all_passed": (nf_cnt == 0 and sim_cnt == 0 and put_cnt == 0),
            "non_finite_count": nf_cnt,
            "simplex_violation_count": sim_cnt,
            "put_bound_violation_count": put_cnt,
        }

    def record_batch_audit(
        self,
        *,
        trade_id: str,
        pre_diag: dict[str, Any],
        post_diag: dict[str, Any],
        engine_selected: str,
        greeks_engine_used: str,
    ) -> SR262AuditRecord:
        """Records a batch SR 26-2 compliance telemetry entry into `self.audit_log`."""
        fb_reasons: list[str] = []
        warnings: list[str] = []
        if pre_diag.get("non_finite_count", 0) > 0 or post_diag.get("non_finite_count", 0) > 0:
            fb_reasons.append(AutocallFallbackReason.NON_FINITE_OUTPUT.value)
        if pre_diag.get("ood_count", 0) > 0:
            warnings.append(f"BATCH_OOD_COUNT={pre_diag['ood_count']}")
        if pre_diag.get("conformal_breach_count", 0) > 0:
            warnings.append(f"BATCH_CONFORMAL_BREACH_COUNT={pre_diag['conformal_breach_count']}")
        if pre_diag.get("psi_breached", False):
            fb_reasons.append(AutocallFallbackReason.PSI_DRIFT_BREACH.value)
        if post_diag.get("simplex_violation_count", 0) > 0:
            fb_reasons.append(AutocallFallbackReason.POST_INFERENCE_SIMPLEX_VIOLATION.value)
        if post_diag.get("put_bound_violation_count", 0) > 0:
            fb_reasons.append(AutocallFallbackReason.POST_INFERENCE_PUT_BOUND_VIOLATION.value)

        rec = SR262AuditRecord(
            trade_id=trade_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            pre_gate_passed=(pre_diag.get("non_finite_count", 0) == 0 and not pre_diag.get("psi_breached", False)),
            post_gate_passed=bool(post_diag.get("all_passed", True)),
            fallback_triggered=bool(len(fb_reasons) > 0),
            fallback_reasons=fb_reasons,
            warnings=warnings,
            min_corr_eigenvalue_raw=1.0,
            min_corr_eigenvalue_shrunk=1.0,
            correlation_shrunk_applied=False,
            mahalanobis_d2=float(pre_diag.get("max_mahalanobis_d2", 0.0)),
            max_psi_score=float(pre_diag.get("max_psi_score", 0.0)),
            min_pin_risk_z=100.0,
            engine_selected=engine_selected,
            conformal_half_width_bps=float(pre_diag.get("max_conformal_half_width_bps", 0.0)),
            greeks_engine_used=greeks_engine_used,
        )
        with self._lock:
            self.audit_log.append(rec)
        return rec

    def price_with_guardian(
        self,
        req_or_state: Any,
        surrogate: MaskedEQAutocallSurrogate | None = None,
        *,
        n_fallback_paths: int = 32768,
        qmc_mode: Literal["dupire_lv", "terminal_iv", "auto"] = "auto",
        n_substeps_per_period: int = 16,
        force_qmc: bool = False,
        record_drift: bool | None = None,
        seed: int | None = None,
        greeks_fn: Any = None,
        greeks_engine_used: str | None = None,
    ) -> GuardianPricingResult:
        """Executes the guarded 4-Tier pricing pipeline with deterministic failover to Dupire QMC."""
        if isinstance(req_or_state, AutocallPreprocState):
            state = req_or_state
        else:
            state = preprocess_eq_autocall(req_or_state, d_max=self.d_max, m_max=self.m_max)

        trade_id = state.trade_info if state.trade_info is not None else "UNSPECIFIED_TRADE"
        ts_utc = datetime.now(timezone.utc).isoformat()
        cash_today = float(
            getattr(
                state,
                "cash_to_settle_today",
                getattr(state.seasoned_state, "cash_to_settle_today", 0.0),
            )
        )

        pre_passed, pre_reasons, r_psd, diag = self.evaluate_pre_inference(
            state,
            state.baseline,
            record_drift=record_drift,
        )

        if diag["correlation_shrunk_applied"] or (
            AutocallFallbackReason.CORRELATION_NOT_PSD in pre_reasons
        ):
            self._apply_shrunk_correlation_to_state(state, r_psd)

        all_reasons: list[AutocallFallbackReason] = list(pre_reasons)
        warnings: list[str] = list(diag["warnings"])
        conformal_hw_bps = float(diag.get("conformal_half_width_bps", 0.0))

        if force_qmc and AutocallFallbackReason.ENGINE_REQUESTED_QMC not in all_reasons:
            all_reasons.append(AutocallFallbackReason.ENGINE_REQUESTED_QMC)

        # Case 1: M_rem == 0 or already autocalled -> Deterministic Tier 1 valuation in 0 ms
        if state.m_rem == 0 or state.already_autocalled:
            m_rem = state.m_rem
            z_t = torch.zeros(m_rem, dtype=torch.float64, device=state.spots_t0.device)
            post_det = assemble_autocall_cashflows_and_pv(
                state,
                p_stop=z_t,
                q_cpn=z_t,
                l_put=torch.tensor(0.0, dtype=torch.float64, device=state.spots_t0.device),
                option_se=0.0,
            )
            eff_greeks_eng = greeks_engine_used or (
                "Tier1_AnalyticalProxy" if greeks_fn is not None else "None"
            )
            audit_det = SR262AuditRecord(
                trade_id=trade_id,
                timestamp_utc=ts_utc,
                pre_gate_passed=False,
                post_gate_passed=True,
                fallback_triggered=True,
                fallback_reasons=[r.value for r in all_reasons],
                warnings=warnings,
                min_corr_eigenvalue_raw=diag["min_corr_eigenvalue_raw"],
                min_corr_eigenvalue_shrunk=diag["min_corr_eigenvalue_shrunk"],
                correlation_shrunk_applied=diag["correlation_shrunk_applied"],
                mahalanobis_d2=diag["mahalanobis_d2"],
                max_psi_score=diag["max_psi_score"],
                min_pin_risk_z=diag["min_pin_risk_z"],
                engine_selected="Tier1_Deterministic",
                conformal_half_width_bps=conformal_hw_bps,
                greeks_engine_used=eff_greeks_eng,
            )
            with self._lock:
                self.audit_log.append(audit_det)
            z_np = np.zeros(m_rem, dtype=np.float64)
            return GuardianPricingResult(
                product_value=float(post_det.product_value.detach().cpu().item()),
                standard_error=0.0,
                notional_pv=float(post_det.notional_pv.detach().cpu().item()),
                coupon_pv=float(post_det.coupon_pv.detach().cpu().item()),
                option_pv=float(post_det.option_pv.detach().cpu().item()),
                funding_pv=float(post_det.funding_pv.detach().cpu().item()),
                pv_in_transit=float(post_det.pv_in_transit.detach().cpu().item()),
                p_stop=z_np,
                q_cpn=z_np,
                l_put=0.0,
                l_call=0.0,
                cashflows=post_det.cashflows,
                reporting_currency=post_det.reporting_currency,
                cash_to_settle_today=post_det.cash_to_settle_today,
                engine_selected="Tier1_Deterministic",
                fallback_triggered=True,
                fallback_reasons=all_reasons,
                audit_record=audit_det,
                shrunk_corr_matrix=r_psd,
                preproc_state=state,
                postproc_out=post_det,
                qmc_result=None,
                surrogate_out=None,
            )

        # Case 2: Pre-Inference Gate passed and Tier 2 Surrogate is available (and not force_qmc)
        post_passed = True
        surrogate_out: AutocallSurrogateOutput | None = None
        post_surr: AutocallPostprocOutput | None = None
        greeks_res: Any = None

        if pre_passed and not force_qmc and surrogate is not None:
            if greeks_fn is not None and callable(greeks_fn):
                greeks_res = greeks_fn(state, surrogate)
                post_surr = greeks_res.postproc_output
                assert post_surr is not None
                p_stop_t = post_surr.p_stop
                q_cpn_t = post_surr.q_cpn
                l_put_t = post_surr.l_put
            else:
                tier2_inputs = state.to_tier2_float32_tensors()
                surr_device = next(surrogate.parameters()).device
                t2_dev = {k: v.to(device=surr_device) for k, v in tier2_inputs.items()}
                with torch.no_grad():
                    surrogate_out = surrogate(
                        node_feats=t2_dev["node_feats"],
                        edge_feats=t2_dev["edge_feats"],
                        asset_mask=t2_dev["asset_mask"],
                        sched_feats=t2_dev["sched_feats"],
                        sched_mask=t2_dev["sched_mask"],
                        global_feats=t2_dev["global_feats"],
                    )

                m_rem = state.m_rem
                s_dev = state.spots_t0.device
                p_stop_t = (
                    surrogate_out.stopping_probs[0, :m_rem]
                    .to(dtype=torch.float64, device=s_dev)
                    * state.is_callable
                )
                q_cpn_t = surrogate_out.coupon_probs[0, :m_rem].to(
                    dtype=torch.float64, device=s_dev
                )
                l_put_t = surrogate_out.put_loss[0].to(dtype=torch.float64, device=s_dev)

                post_surr = assemble_autocall_cashflows_and_pv(
                    state,
                    p_stop=p_stop_t,
                    q_cpn=q_cpn_t,
                    l_put=l_put_t,
                    option_se=0.0,
                )

            post_passed, post_reasons = self.evaluate_post_inference(
                surrogate_out,
                post_surr,
                state,
                state.baseline,
            )
            if post_passed:
                eff_greeks_t2 = greeks_engine_used or (
                    "Tier2_SingleVJP_AD" if greeks_res is not None else "None"
                )
                audit_t2 = SR262AuditRecord(
                    trade_id=trade_id,
                    timestamp_utc=ts_utc,
                    pre_gate_passed=True,
                    post_gate_passed=True,
                    fallback_triggered=False,
                    fallback_reasons=[],
                    warnings=warnings,
                    min_corr_eigenvalue_raw=diag["min_corr_eigenvalue_raw"],
                    min_corr_eigenvalue_shrunk=diag["min_corr_eigenvalue_shrunk"],
                    correlation_shrunk_applied=diag["correlation_shrunk_applied"],
                    mahalanobis_d2=diag["mahalanobis_d2"],
                    max_psi_score=diag["max_psi_score"],
                    min_pin_risk_z=diag["min_pin_risk_z"],
                    engine_selected="Tier2_MaskedEGNO",
                    conformal_half_width_bps=conformal_hw_bps,
                    greeks_engine_used=eff_greeks_t2,
                )
                with self._lock:
                    self.audit_log.append(audit_t2)
                return GuardianPricingResult(
                    product_value=float(post_surr.product_value.detach().cpu().item()),
                    standard_error=0.0,
                    notional_pv=float(post_surr.notional_pv.detach().cpu().item()),
                    coupon_pv=float(post_surr.coupon_pv.detach().cpu().item()),
                    option_pv=float(post_surr.option_pv.detach().cpu().item()),
                    funding_pv=float(post_surr.funding_pv.detach().cpu().item()),
                    pv_in_transit=float(post_surr.pv_in_transit.detach().cpu().item()),
                    p_stop=p_stop_t.detach().cpu().numpy().astype(np.float64),
                    q_cpn=q_cpn_t.detach().cpu().numpy().astype(np.float64),
                    l_put=float(l_put_t.detach().cpu().item()),
                    l_call=float(post_surr.l_call.detach().cpu().item()),
                    cashflows=post_surr.cashflows,
                    reporting_currency=post_surr.reporting_currency,
                    cash_to_settle_today=post_surr.cash_to_settle_today,
                    engine_selected="Tier2_MaskedEGNO",
                    fallback_triggered=False,
                    fallback_reasons=[],
                    audit_record=audit_t2,
                    shrunk_corr_matrix=r_psd,
                    preproc_state=state,
                    postproc_out=post_surr,
                    qmc_result=None,
                    surrogate_out=surrogate_out,
                    greeks_out=greeks_res,
                )

            for r in post_reasons:
                if r not in all_reasons:
                    all_reasons.append(r)

        # Case 3: Route to Tier 3/4 Deterministic Dupire Local Volatility RQMC Fallback
        if surrogate is None and len(all_reasons) == 0:
            all_reasons.append(AutocallFallbackReason.ENGINE_REQUESTED_QMC)

        idx_set = {
            str(state.callability_index).strip(),
            str(state.coupon_index or state.callability_index).strip(),
            str(state.option_index or state.callability_index).strip(),
        }
        if self.use_gpu_qmc is not None:
            eff_use_gpu = bool(self.use_gpu_qmc) and torch.cuda.is_available()
        else:
            eff_use_gpu = (
                torch.cuda.is_available()
                and int(n_fallback_paths) >= 1024
                and ("TwinWin" not in idx_set)
            )

        qmc_res = price_eq_autocall_dupire_qmc(
            state,
            n_paths=n_fallback_paths,
            mode=qmc_mode,
            n_substeps_per_period=n_substeps_per_period,
            seed=seed,
            use_gpu=eff_use_gpu,
        )

        eff_greeks_qmc = greeks_engine_used or "None"
        audit_fb = SR262AuditRecord(
            trade_id=trade_id,
            timestamp_utc=ts_utc,
            pre_gate_passed=pre_passed,
            post_gate_passed=post_passed,
            fallback_triggered=True,
            fallback_reasons=[r.value for r in all_reasons],
            warnings=warnings,
            min_corr_eigenvalue_raw=diag["min_corr_eigenvalue_raw"],
            min_corr_eigenvalue_shrunk=diag["min_corr_eigenvalue_shrunk"],
            correlation_shrunk_applied=diag["correlation_shrunk_applied"],
            mahalanobis_d2=diag["mahalanobis_d2"],
            max_psi_score=diag["max_psi_score"],
            min_pin_risk_z=diag["min_pin_risk_z"],
            engine_selected="Tier3_DupireQMC",
            conformal_half_width_bps=conformal_hw_bps,
            greeks_engine_used=eff_greeks_qmc,
        )
        with self._lock:
            self.audit_log.append(audit_fb)

        return GuardianPricingResult(
            product_value=float(qmc_res.product_value),
            standard_error=float(qmc_res.standard_error),
            notional_pv=float(qmc_res.notional_pv),
            coupon_pv=float(qmc_res.coupon_pv),
            option_pv=float(qmc_res.option_pv),
            funding_pv=float(qmc_res.funding_pv),
            pv_in_transit=float(state.pv_in_transit),
            p_stop=qmc_res.p_stop,
            q_cpn=qmc_res.q_cpn,
            l_put=float(qmc_res.l_put),
            l_call=float(qmc_res.l_call),
            cashflows=qmc_res.cashflows,
            reporting_currency=state.payment_currency,
            cash_to_settle_today=cash_today,
            engine_selected="Tier3_DupireQMC",
            fallback_triggered=True,
            fallback_reasons=all_reasons,
            audit_record=audit_fb,
            shrunk_corr_matrix=r_psd,
            preproc_state=state,
            postproc_out=None,
            qmc_result=qmc_res,
            surrogate_out=surrogate_out,
        )


__all__ = [
    "DEFAULT_GUARDIAN_STATS_PATH",
    "DEFAULT_TAU_DESK_BPS",
    "SUPPORTED_TIER2_INDICES",
    "CHI2_16_0999",
    "AutocallFallbackReason",
    "AutocallSummaryCol",
    "FEATURE_NAMES_16D",
    "SR262AuditRecord",
    "GuardianPricingResult",
    "project_correlation_to_psd",
    "extract_16d_basket_summary",
    "compute_psi_10_decile",
    "EQAutocallModelGuardian",
]
