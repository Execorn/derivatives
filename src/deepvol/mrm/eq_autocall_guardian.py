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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
import logging
import math
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
from deepvol.models.autocall.dupire_qmc_solver import (
    DupireQMCResult,
    price_eq_autocall_dupire_qmc,
)
from deepvol.models.autocall.pre_post_processor import (
    D_MAX_DEFAULT,
    M_MAX_DEFAULT,
    AutocallEdgeCol,
    AutocallNodeCol,
    AutocallPostprocOutput,
    AutocallPreprocState,
    AutocallSchedCol,
    assemble_autocall_cashflows_and_pv,
    preprocess_eq_autocall,
)
from deepvol.models.autocall.seasoned_collapser import CashFlowItem
from deepvol.surrogates.eq_autocall_egno import (
    AutocallSurrogateOutput,
    MaskedEQAutocallSurrogate,
)

logger = logging.getLogger("deepvol.mrm.eq_autocall_guardian")

SUPPORTED_TIER2_INDICES: frozenset[str] = frozenset(
    {"WorstOf", "BestOf", "Average", "AverageOf"}
)

CHI2_16_0999: float = 39.25


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

    def to_dict(self) -> dict[str, Any]:
        """Serializes the audit record to a JSON-compatible dictionary."""
        return asdict(self)


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
    """Projects a symmetric correlation matrix R to the SPD cone with lambda_min >= eps_psd.

    Applies optimal Ledoit-Wolf / convex shrinkage toward the identity matrix:
        R(alpha) = (1 - alpha) * R_sym + alpha * I_D,
        alpha = clip((eps_psd - lambda_min) / (1 - lambda_min), 0.0, 1.0),
    followed by spectral eigenvalue flooring + diagonal renormalization if needed so that
    lambda_min(R_PSD) >= eps_psd and diag(R_PSD) == 1 in IEEE-754 float64.

    Args:
        corr: Correlation matrix of shape (D, D) as numpy array, torch.Tensor, or nested list.
        eps_psd: Minimum allowable eigenvalue threshold (default 1e-4).

    Returns:
        r_psd: (D, D) float64 PSD correlation matrix with unit diagonal and lambda_min >= eps_psd.
        min_eig_raw: Minimum eigenvalue of the raw symmetrized input matrix.
        min_eig_shrunk: Minimum eigenvalue of the returned R_PSD matrix.
        shrunk_applied: True if shrinkage or projection was applied, else False.
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
        r_sym = 0.5 * (r_clean + r_clean.T)
        min_eig_raw = -1.0
    else:
        r_sym = 0.5 * (r_in + r_in.T)
        eigvals_raw = np.linalg.eigvalsh(r_sym)
        min_eig_raw = float(np.min(eigvals_raw))

    diag_dev = float(np.max(np.abs(np.diag(r_sym) - 1.0)))
    off_mask = ~np.eye(d, dtype=bool)
    max_off = float(np.max(np.abs(r_sym[off_mask]))) if np.any(off_mask) else 0.0

    if (
        not has_non_finite
        and min_eig_raw >= eps_psd
        and diag_dev <= 1e-10
        and max_off <= 1.0 - 1e-12
    ):
        np.fill_diagonal(r_sym, 1.0)
        return r_sym, min_eig_raw, min_eig_raw, False

    # Ensure unit diagonal and bounded off-diagonal entries before convex shrinkage
    r_work = r_sym.copy()
    np.fill_diagonal(r_work, 1.0)
    max_allowed_off = 1.0 - eps_psd
    if max_off > max_allowed_off:
        r_work[off_mask] = np.clip(r_work[off_mask], -max_allowed_off, max_allowed_off)

    eigvals_work = np.linalg.eigvalsh(r_work)
    lam_min = float(np.min(eigvals_work))

    if lam_min < eps_psd:
        denom = max(1.0 - lam_min, 1e-12)
        alpha = min(max((eps_psd - lam_min) / denom, 0.0), 1.0)
        # Cap global convex shrinkage at 0.05 so spectral projection handles localized negative modes
        # without shrinking valid off-diagonal correlations by up to 46.4% (AUD-P1-06)
        alpha_eff = min(alpha, 0.05) if not has_non_finite else alpha
        eye_d = np.eye(d, dtype=np.float64)
        r_psd = (1.0 - alpha_eff) * r_work + alpha_eff * eye_d
        np.fill_diagonal(r_psd, 1.0)
    else:
        r_psd = r_work

    # Verify minimum eigenvalue after convex shrinkage; apply spectral floor + renormalization if needed
    eigvals_post, eigvecs_post = np.linalg.eigh(r_psd)
    min_eig_shrunk = float(np.min(eigvals_post))
    if min_eig_shrunk < eps_psd - 1e-14:
        eigvals_floored = np.maximum(eigvals_post, eps_psd)
        r_spec = (eigvecs_post * eigvals_floored[None, :]) @ eigvecs_post.T
        d_inv = 1.0 / np.sqrt(np.maximum(np.diag(r_spec), 1e-12))
        r_psd = r_spec * np.outer(d_inv, d_inv)
        r_psd = 0.5 * (r_psd + r_psd.T)
        np.fill_diagonal(r_psd, 1.0)
        eigvals_final = np.linalg.eigvalsh(r_psd)
        lam_final = float(np.min(eigvals_final))
        if lam_final < eps_psd:
            alpha_fix = min(max((eps_psd - lam_final + 1e-12) / max(1.0 - lam_final, 1e-12), 0.0), 1.0)
            r_psd = (1.0 - alpha_fix) * r_psd + alpha_fix * np.eye(d, dtype=np.float64)
            np.fill_diagonal(r_psd, 1.0)
        min_eig_shrunk = float(np.min(np.linalg.eigvalsh(r_psd)))

    return r_psd, min_eig_raw, min_eig_shrunk, True


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
    """Computes the 10-decile Population Stability Index (PSI) across the 16 summary features.

    Formula:
        PSI_j = sum_{b=1}^{10} (p_{j,b} - q_{j,b}) * ln(max(p_{j,b}, 1e-6) / max(q_{j,b}, 1e-6))

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
            counts, _ = np.histogram(col_finite, bins=bin_edges[j])
            p_b = counts.astype(np.float64) / float(len(col_finite))
            q_b = ref_probs[j]
            p_safe = np.maximum(p_b, 1e-6)
            q_safe = np.maximum(q_b, 1e-6)
            psi_j = float(np.sum((p_b - q_b) * np.log(p_safe / q_safe)))
        name_j = FEATURE_NAMES_16D[j]
        psi_map[name_j] = psi_j
        if psi_j > max_psi:
            max_psi = psi_j

    return max_psi, psi_map


class EQAutocallModelGuardian:
    """Tier 3 SR 11-7 / SR 26-2 Model Risk Management Guardian & Failover Router.

    Enforces a 3-stage circuit breaker:
      1. Pre-Inference Structural, PSD Correlation, Imminent Pin-Risk, Mahalanobis OOD & PSI Gate.
      2. Post-Inference Probability Simplex, Put Loss Bound & Residual Outlier Gate.
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
        residual_threshold: float = 0.15,
        ref_mean_16d: npt.NDArray[np.float64] | None = None,
        ref_cov_inv_16d: npt.NDArray[np.float64] | None = None,
        record_drift_by_default: bool = False,
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
        self.residual_threshold = float(residual_threshold)
        self.record_drift_by_default = bool(record_drift_by_default)

        self.ref_mean_16d: npt.NDArray[np.float64] = (
            np.asarray(ref_mean_16d, dtype=np.float64).copy()
            if ref_mean_16d is not None
            else _DEFAULT_REF_MEAN_16D.copy()
        )
        if ref_cov_inv_16d is not None:
            self.ref_cov_inv_16d: npt.NDArray[np.float64] = np.asarray(
                ref_cov_inv_16d, dtype=np.float64
            ).copy()
            ref_std = np.sqrt(
                1.0 / np.maximum(np.diag(self.ref_cov_inv_16d), 1e-12)
            )
        else:
            ref_std = _DEFAULT_REF_STD_16D.copy()
            self.ref_cov_inv_16d = np.diag(1.0 / (ref_std * ref_std))

        # Initialize 10-decile bin edges from the calibrated Gaussian reference distribution
        self.psi_bin_edges = np.zeros((16, 11), dtype=np.float64)
        self.psi_ref_probs = np.full((16, 10), 0.10, dtype=np.float64)
        inner_z = np.asarray(norm.ppf(np.linspace(0.1, 0.9, 9)), dtype=np.float64)
        for j in range(16):
            self.psi_bin_edges[j, 0] = -np.inf
            self.psi_bin_edges[j, 1:10] = self.ref_mean_16d[j] + ref_std[j] * inner_z
            self.psi_bin_edges[j, 10] = np.inf

        self.feature_history: deque[npt.NDArray[np.float64]] = deque(
            maxlen=self.psi_window_size
        )
        self.audit_log: deque[SR262AuditRecord] = deque(maxlen=10_000)
        self._lock = threading.Lock()

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
        self.ref_mean_16d = np.mean(x_ref, axis=0)
        diff = x_ref - self.ref_mean_16d[None, :]
        cov_sample = (diff.T @ diff) / float(max(n_samples - 1, 1))
        diag_target = np.diag(np.maximum(np.diag(cov_sample), 1e-4))
        cov_shrunk = (
            (1.0 - shrinkage) * cov_sample
            + shrinkage * diag_target
            + 1e-4 * np.eye(16, dtype=np.float64)
        )
        self.ref_cov_inv_16d = np.linalg.inv(cov_shrunk)

        percentiles = np.linspace(0.0, 100.0, 11)
        for j in range(16):
            edges = np.percentile(x_ref[:, j], percentiles).astype(np.float64)
            if len(np.unique(edges[1:-1])) < 9:
                col_std = max(float(np.std(x_ref[:, j])), 1e-3)
                inner = self.ref_mean_16d[j] + col_std * np.asarray(
                    norm.ppf(np.linspace(0.1, 0.9, 9)), dtype=np.float64
                )
            else:
                inner = edges[1:-1]
            self.psi_bin_edges[j, 0] = -np.inf
            self.psi_bin_edges[j, 1:10] = inner
            self.psi_bin_edges[j, 10] = np.inf
            counts, _ = np.histogram(x_ref[:, j], bins=self.psi_bin_edges[j])
            self.psi_ref_probs[j] = np.maximum(
                counts.astype(np.float64) / float(max(n_samples, 1)), 1e-6
            )
            self.psi_ref_probs[j] /= np.sum(self.psi_ref_probs[j])

    def reset_drift_history(self) -> None:
        """Clears the rolling feature history window."""
        with self._lock:
            self.feature_history.clear()

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
    ) -> float:
        """Computes squared Mahalanobis distance D_M^2(x) = (x - mu)^T Sigma^{-1} (x - mu) in float64."""
        x = np.asarray(feat_16d, dtype=np.float64).reshape(16)
        if not np.all(np.isfinite(x)):
            return 1e6
        diff = x - self.ref_mean_16d
        d2 = float(diff @ self.ref_cov_inv_16d @ diff)
        return max(d2, 0.0)

    def compute_psi(
        self,
        observed_batch: npt.NDArray[np.float64] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Computes the 10-decile PSI on `observed_batch` or the rolling `self.feature_history` window."""
        if observed_batch is not None:
            obs = np.asarray(observed_batch, dtype=np.float64)
            return compute_psi_10_decile(obs, self.psi_bin_edges, self.psi_ref_probs)
        with self._lock:
            if len(self.feature_history) < self.psi_min_samples:
                return 0.0, {name: 0.0 for name in FEATURE_NAMES_16D}
            obs = np.stack(list(self.feature_history), axis=0)
        return compute_psi_10_decile(obs, self.psi_bin_edges, self.psi_ref_probs)

    def compute_min_pin_risk_z(
        self,
        preproc_state: AutocallPreprocState,
        baseline_out: AutocallBaselineOutput | None = None,
    ) -> tuple[float, bool]:
        """Computes imminent barrier pin-risk z-score across short-dated active periods (tau_m <= 0.08).

        Screens all active barriers (AutoCall B_KO, non-guaranteed Coupon B_cpn, and terminal KnockIn B_KI)
        when tau_m <= self.pin_risk_tau_max (§1.5.1 & AUD-P1-06).

        Returns:
            min_z: Finite float64 pin-risk z-score in [0.0, 100.0].
            pin_risk_breached: True if min_z < self.pin_risk_z_min (0.15) on an imminent barrier period.
        """
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

        min_imminent_z = 100.0
        has_imminent = False
        for m in range(m_rem):
            if tau_np[m] > self.pin_risk_tau_max:
                continue
            denom = max(float(vw_np[m]), 1e-12)
            fw_m = float(fw_np[m])

            # 1. Active AutoCall barrier
            if (call_np[m] > 0.5) or (ko_np[m] < 10.0):
                has_imminent = True
                z_ko = abs(fw_m - float(ko_np[m])) / denom
                if not np.isfinite(z_ko):
                    z_ko = 0.0
                if z_ko < min_imminent_z:
                    min_imminent_z = z_ko

            # 2. Active non-guaranteed Coupon barrier
            if guar_np[m] < 0.5 and 1e-4 < float(cpn_np[m]) < 10.0:
                has_imminent = True
                z_cpn = abs(fw_m - float(cpn_np[m])) / denom
                if not np.isfinite(z_cpn):
                    z_cpn = 0.0
                if z_cpn < min_imminent_z:
                    min_imminent_z = z_cpn

            # 3. Active terminal KnockIn barrier on final observation period
            if (
                m == m_rem - 1
                and preproc_state.ki_barrier is not None
                and float(preproc_state.ki_barrier) > 1e-4
                and float(preproc_state.i_ki_past) < 0.5
            ):
                has_imminent = True
                z_ki = abs(fw_m - float(preproc_state.ki_barrier)) / denom
                if not np.isfinite(z_ki):
                    z_ki = 0.0
                if z_ki < min_imminent_z:
                    min_imminent_z = z_ki

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
        """Executes the Tier 3 Pre-Inference Structural, PSD, Pin-Risk, OOD & PSI Gate in float64.

        Args:
            preproc_state: Tier 1 `AutocallPreprocState` in `torch.float64`.
            baseline_out: Optional `AutocallBaselineOutput` (defaults to `preproc_state.baseline`).
            record_drift: Whether to append this trade's 16D feature summary to `self.feature_history`.

        Returns:
            passed: True if the trade is safe for Tier 2 neural surrogate inference.
            reasons: List of `AutocallFallbackReason` codes that tripped.
            r_psd: (D, D) PSD-shrunk correlation matrix in `np.float64` with lambda_min >= 1e-4.
            diagnostics: Dictionary of SR 26-2 diagnostic metrics and warnings.
        """
        b_out = baseline_out if baseline_out is not None else preproc_state.baseline
        reasons: list[AutocallFallbackReason] = []
        warnings: list[str] = []

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

        # 3. Exotic monitoring index & exotic terminal modifier gates
        indices_to_check = [preproc_state.callability_index]
        if preproc_state.coupon_index is not None:
            indices_to_check.append(preproc_state.coupon_index)
        if preproc_state.option_index is not None:
            indices_to_check.append(preproc_state.option_index)

        for idx_name in indices_to_check:
            if str(idx_name).strip() not in SUPPORTED_TIER2_INDICES:
                reasons.append(AutocallFallbackReason.EXOTIC_MONITORING_INDEX)
                break

        if (
            float(preproc_state.participation_call_rate) > 0.0
            or float(preproc_state.one_star_coupon) > 0.0
            or (preproc_state.one_star_strike is not None and float(preproc_state.one_star_strike) > 0.0)
            or str(preproc_state.coupon_barrier_type).strip() != "In"
            or str(preproc_state.coupon_barrier_direction).strip() != "Up"
        ):
            reasons.append(AutocallFallbackReason.EXOTIC_TERMINAL_MODIFIER_ROUTED)

        # 4. Correlation PSD & Ledoit-Wolf shrinkage gate
        r_psd, min_eig_raw, min_eig_shrunk, shrunk_applied = project_correlation_to_psd(
            preproc_state.corr_matrix,
            eps_psd=self.eps_psd,
        )
        raw_corr_np = preproc_state.corr_matrix.detach().cpu().numpy().astype(np.float64)
        off_mask = ~np.eye(d_assets, dtype=bool) if d_assets >= 2 else np.zeros((d_assets, d_assets), dtype=bool)
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

        # 5. Imminent barrier pin-risk gate
        min_pin_z, pin_breached = self.compute_min_pin_risk_z(preproc_state, b_out)
        if pin_breached:
            reasons.append(AutocallFallbackReason.IMMINENT_BARRIER_PIN_RISK)

        # 6. 16D Basket Summary Feature Extraction, Mahalanobis OOD & 10-Decile PSI Drift Gate
        summary_16d = extract_16d_basket_summary(preproc_state, b_out)
        if not bool(np.all(np.isfinite(summary_16d))):
            if AutocallFallbackReason.NON_FINITE_OUTPUT not in reasons:
                reasons.append(AutocallFallbackReason.NON_FINITE_OUTPUT)
            mahalanobis_d2 = 1e6
        else:
            mahalanobis_d2 = self.compute_mahalanobis_d2(summary_16d)
            # Enforce explicit Sobol training-domain box bounds (AUD-P0-09)
            max_vol_val = float(summary_16d[AutocallSummaryCol.MAX_ATM_VOL])
            max_div_lev = float(summary_16d[AutocallSummaryCol.MAX_CASH_DIV_LEVERAGE])
            min_log_m = float(summary_16d[AutocallSummaryCol.MIN_LOG_MONEYNESS])
            max_log_m = float(summary_16d[AutocallSummaryCol.MAX_LOG_MONEYNESS])
            if (
                max_vol_val > 1.00
                or max_div_lev > math.log(1.0 / (1.0 - 0.25))
                or min_log_m < math.log(0.15)
                or max_log_m > math.log(2.50)
            ):
                mahalanobis_d2 = max(mahalanobis_d2, self.mahalanobis_chi2_threshold + 10.0)

        if mahalanobis_d2 > self.mahalanobis_chi2_threshold:
            reasons.append(AutocallFallbackReason.MAHALANOBIS_OOD)

        do_record = (
            self.record_drift_by_default if record_drift is None else bool(record_drift)
        )
        if do_record and bool(np.all(np.isfinite(summary_16d))):
            self.record_feature_vector(summary_16d)

        max_psi, psi_per_feature = self.compute_psi()
        if max_psi >= self.psi_fallback_threshold:
            reasons.append(AutocallFallbackReason.PSI_DRIFT_BREACH)
            warnings.append(
                f"PSI_DRIFT_BREACH: max_psi={max_psi:.4f} >= {self.psi_fallback_threshold:.4f}."
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
        """Executes the Tier 3 Post-Inference Probability Simplex, Put Bound & Residual Gate in float64.

        Checks (§1.5.2):
          1. `NON_FINITE_OUTPUT`: No NaN or Inf in p_m, q_m, L_put, or ProductValue.
          2. `POST_INFERENCE_SIMPLEX_VIOLATION`:
             sum_{m=1}^{M_rem} p_m <= 1 + 1e-6, p_m in [-1e-7, 1 + 1e-7], q_m in [-1e-7, 1 + 1e-7].
          3. `POST_INFERENCE_PUT_BOUND_VIOLATION`:
             L_put in [-1e-7, K_put * S_{M_rem} + 1e-6].
          4. `POST_INFERENCE_RESIDUAL_OUTLIER`:
             |PV_surrogate - PV_proxy| <= tau_res (dispersion- and variance-scaled residual bound, AUD-P0-10).
        """
        reasons: list[AutocallFallbackReason] = []
        m_rem = int(preproc_state.m_rem)
        b_out = baseline_out if baseline_out is not None else preproc_state.baseline

        # Extract p_m, q_m, l_put in float64
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

        # If postproc_out was not supplied, assemble it to evaluate PV_surrogate
        if pv_surrogate is None:
            if np.all(np.isfinite(p_arr)) and np.all(np.isfinite(q_arr)) and np.isfinite(l_put_val):
                asm = assemble_autocall_cashflows_and_pv(
                    preproc_state,
                    p_stop=torch.as_tensor(p_arr, dtype=torch.float64),
                    q_cpn=torch.as_tensor(q_arr, dtype=torch.float64),
                    l_put=torch.tensor(l_put_val, dtype=torch.float64),
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

        # 3. Terminal unit put loss bound check: 0 <= L_put <= K_put * S_{M_rem}
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
        max_put_bound = k_put * surv_term + 1e-6
        if l_put_val < -1e-7 or l_put_val > max_put_bound:
            reasons.append(AutocallFallbackReason.POST_INFERENCE_PUT_BOUND_VIOLATION)

        # 4. Residual outlier check: |PV_surrogate - PV_proxy| <= tau_res (AUD-P0-10)
        pv_proxy = float(b_out.product_value_proxy.detach().cpu().item())
        if abs(self.residual_threshold - 0.15) < 1e-9:
            w_np = b_out.proxy_delta_weights.detach().cpu().numpy().astype(np.float64)
            w_sum = max(float(np.sum(np.maximum(w_np, 0.0))), 1e-6)
            w_norm = np.maximum(w_np, 0.0) / w_sum
            h_disp = float(np.clip(1.0 - np.sum(w_norm * w_norm), 0.0, 1.0))
            v_bar = (
                float(np.mean(b_out.v_w.detach().cpu().numpy().astype(np.float64)))
                if m_rem > 0
                else 0.0
            )
            tau_res = min(0.0050 + 0.0350 * h_disp * math.sqrt(max(v_bar, 1e-4)), 0.0200)
        else:
            tau_res = self.residual_threshold
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
        return new_baseline

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
    ) -> GuardianPricingResult:
        """Executes the guarded 4-Tier pricing pipeline with deterministic failover to Dupire QMC.

        Flow:
          1. Runs Tier 1 `preprocess_eq_autocall` (if not already `AutocallPreprocState`) + Stulz-Margrabe baseline.
          2. Evaluates Tier 3 Pre-Inference Gate (`evaluate_pre_inference`) and applies PSD correlation shrinkage.
          3. If `m_rem == 0` (`ZERO_FUTURE_PERIODS`): assembles deterministic cashflows and PV in Tier 1 (0 ms).
          4. If Pre-Inference Gate passes and `surrogate is not None` and not `force_qmc`:
             runs Tier 2 `MaskedEQAutocallSurrogate` + Tier 1 W1 cash-flow assembly + Post-Inference Gate.
          5. If either gate trips (or `force_qmc=True` / `surrogate is None`):
             routes seamlessly to `price_eq_autocall_dupire_qmc` using the PSD-shrunk correlation matrix
             and logs a complete `SR262AuditRecord` in `self.audit_log`.
        """
        if isinstance(req_or_state, AutocallPreprocState):
            state = req_or_state
        else:
            state = preprocess_eq_autocall(req_or_state, d_max=self.d_max, m_max=self.m_max)

        trade_id = state.trade_info if state.trade_info is not None else "UNSPECIFIED_TRADE"
        ts_utc = datetime.now(timezone.utc).isoformat()

        pre_passed, pre_reasons, r_psd, diag = self.evaluate_pre_inference(
            state,
            state.baseline,
            record_drift=record_drift,
        )

        # Always ensure downstream state carries the valid PSD correlation matrix
        if diag["correlation_shrunk_applied"] or (
            AutocallFallbackReason.CORRELATION_NOT_PSD in pre_reasons
        ):
            self._apply_shrunk_correlation_to_state(state, r_psd)

        all_reasons: list[AutocallFallbackReason] = list(pre_reasons)
        warnings: list[str] = list(diag["warnings"])

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

        # Case 3: Route to Tier 3 Deterministic Dupire Local Volatility Sobol QMC Fallback
        if surrogate is None and len(all_reasons) == 0:
            all_reasons.append(AutocallFallbackReason.ENGINE_REQUESTED_QMC)

        qmc_res = price_eq_autocall_dupire_qmc(
            state,
            n_paths=n_fallback_paths,
            mode=qmc_mode,
            n_substeps_per_period=n_substeps_per_period,
            seed=seed,
        )

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
            cash_to_settle_today=0.0,
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
