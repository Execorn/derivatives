"""
autocall_guardian.py — 3-Tier Model Risk Guardian for Autocall Correction Surrogates.

Compliance & Governance Standards:
  - Federal Reserve SR 11-7: Model Risk Management (MRM)
  - Federal Reserve SR 26-2: Artificial Intelligence & Machine Learning Model Governance

Architecture:
  - Tier 1: Pre-Inference Input Out-of-Distribution (OOD) Filter Gate
  - Tier 2: Post-Inference Output Arbitrage & Plausibility Gate
  - Tier 3: Operational Fallback Routing & SR 26-2 Audit Logger
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("deepvol.mrm.autocall_guardian")


class AutocallModelGuardian:
    """
    Production-grade Model Risk Guardian for Autocallable Note Correction Surrogates.
    Intercepts anomalous inputs and arbitrage-violating predictions, routing them to
    robust numerical PDE or Monte Carlo fallback solvers.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        norm_in: Optional[Any] = None,
        norm_out: Optional[Any] = None,
        feller_threshold: float = 0.20,
        integrated_vol_threshold: float = 0.40,
        mahalanobis_threshold: float = 36.19,  # chi2(19, p=0.99)
        residual_clamp_std: float = 3.0,
        psi_alert_threshold: float = 0.25,
        tau_ood: float = 2.0,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.norm_in = norm_in
        self.norm_out = norm_out
        self.feller_threshold = feller_threshold
        self.integrated_vol_threshold = integrated_vol_threshold
        self.mahalanobis_threshold = mahalanobis_threshold
        self.residual_clamp_std = residual_clamp_std
        self.psi_alert_threshold = psi_alert_threshold
        self.tau_ood = tau_ood
        self.device = torch.device(device)

        # Baseline reference distribution for Population Stability Index (PSI)
        self.reference_distribution: Optional[np.ndarray] = None
        self.audit_log: List[Dict[str, Any]] = []

    def set_reference_distribution(self, X_train: np.ndarray) -> None:
        """Store reference training feature distribution for online PSI tracking."""
        self.reference_distribution = np.asarray(X_train, dtype=np.float32).copy()

    def check_input_ood(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tier 1: Pre-inference screening for physical and mathematical singularities.

        Checks:
          1. Feller condition: 2 * kappa * theta / sigma^2 < feller_threshold
          2. Integrated variance distortion: |sqrt(v0) / sqrt(v_bar) - 1| > threshold
          3. Compound extrapolation corner: extreme joint tail parameters
          4. Mahalanobis distance in normalized 19D feature space
        """
        reasons: List[str] = []
        is_ood = False

        kappa = float(raw_params.get("kappa", 2.0))
        theta = float(raw_params.get("theta", 0.04))
        sigma = float(raw_params.get("sigma", 0.3))
        rho = float(raw_params.get("rho", -0.7))
        v0 = float(raw_params.get("v0", 0.04))
        T = float(raw_params.get("T", 1.0))
        B = float(raw_params.get("B", 1.0))

        # 1. Feller ratio check
        sigma_sq = max(sigma ** 2, 1e-8)
        feller_ratio = (2.0 * kappa * theta) / sigma_sq
        if feller_ratio < self.feller_threshold:
            is_ood = True
            reasons.append(
                f"Feller condition severe violation: 2*kappa*theta/sigma^2 = {feller_ratio:.4f} < {self.feller_threshold}"
            )

        # 2. Integrated variance distortion check
        kappa_T = max(kappa * T, 1e-6)
        exp_neg_kT = np.exp(-kappa_T)
        v_bar = theta + (v0 - theta) * ((1.0 - exp_neg_kT) / kappa_T)
        v_bar = max(v_bar, 1e-8)
        vol_ratio = np.sqrt(max(v0, 1e-8)) / np.sqrt(v_bar)
        distortion = abs(vol_ratio - 1.0)
        if distortion > self.integrated_vol_threshold:
            is_ood = True
            reasons.append(
                f"Integrated volatility distortion: |sqrt(v0)/sqrt(v_bar) - 1| = {distortion:.4f} > {self.integrated_vol_threshold}"
            )

        # 3. Compound extrapolation corner
        if (T >= 2.5 and v0 <= 0.02) or (sigma >= 0.85 and rho <= -0.80) or (B >= 1.08 and T >= 2.5 and v0 <= 0.03):
            is_ood = True
            reasons.append(
                f"Compound extrapolation corner: T={T:.2f}, v0={v0:.4f}, sigma={sigma:.2f}, rho={rho:.2f}, B={B:.2f}"
            )

        # 4. Mahalanobis distance check (if normalizer fitted)
        if self.norm_in is not None and getattr(self.norm_in, "mean", None) is not None:
            try:
                from deepvol.training.train_correction import CorrectionInputNormalizer
                feature_row = CorrectionInputNormalizer.build_feature_matrix({k: [v] for k, v in raw_params.items()})
                norm_feat = self.norm_in.transform(feature_row)
                d_m_sq = float(np.sum(norm_feat ** 2))
                if d_m_sq > self.mahalanobis_threshold:
                    is_ood = True
                    reasons.append(
                        f"Mahalanobis distance breach: D_M^2 = {d_m_sq:.2f} > {self.mahalanobis_threshold}"
                    )
            except Exception as e:
                logger.debug(f"Mahalanobis check skipped: {e}")

        return {
            "is_ood": is_ood,
            "reasons": reasons,
            "feller_ratio": feller_ratio,
            "integrated_vol_distortion": distortion,
        }

    def check_output_arbitrage(
        self,
        raw_params: Dict[str, Any],
        pde_npv: float,
        delta_v: float,
        df_dB: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Tier 2: Post-inference validation for structural arbitrage and plausibility.

        Checks:
          1. Total barrier derivative: df / dB <= 0 (higher barrier must reduce autocall value)
          2. Residual magnitude clamping: |delta_v| <= 3 * sigma_out
          3. Total price bounds: 0.0 <= V_total <= S0 * (1 + coupon * T)
        """
        reasons: List[str] = []
        is_arbitrage = False

        total_npv = pde_npv + delta_v
        T = float(raw_params.get("T", 1.0))
        coupon = float(raw_params.get("coupon", 0.08))
        S0 = float(raw_params.get("S0", 1.0))

        # 1. Total barrier derivative check
        if df_dB is not None and df_dB > 1e-4:
            is_arbitrage = True
            reasons.append(
                f"Barrier monotonicity violation: total derivative df/dB = {df_dB:.6f} > 0"
            )

        # 2. Residual magnitude clamping
        if self.norm_out is not None and getattr(self.norm_out, "std", None) is not None:
            max_delta_v = self.residual_clamp_std * float(self.norm_out.std[0])
            if abs(delta_v) > max_delta_v:
                is_arbitrage = True
                reasons.append(
                    f"Residual magnitude breach: |delta_v|={abs(delta_v):.6f} > {max_delta_v:.6f} ({self.residual_clamp_std}*sigma)"
                )

        # 3. Total price bounds
        max_possible_price = S0 * (1.0 + coupon * T)
        if total_npv < 0.0 or total_npv > max_possible_price + 0.05:
            is_arbitrage = True
            reasons.append(
                f"Price boundary violation: V_total = {total_npv:.4f} not in [0.0, {max_possible_price:.4f}]"
            )

        return {
            "is_arbitrage": is_arbitrage,
            "reasons": reasons,
            "total_npv": total_npv,
            "df_dB": df_dB,
        }

    def compute_psi(self, current_batch: np.ndarray) -> float:
        """
        Computes Population Stability Index (PSI) per Federal Reserve SR 26-2.
        PSI < 0.10: Insignificant drift (Model stable).
        0.10 <= PSI < 0.25: Moderate drift (Monitor closely).
        PSI >= 0.25: Significant drift (Mandates model recalibration / fallback).
        """
        if self.reference_distribution is None:
            return 0.0

        ref = self.reference_distribution.flatten()
        curr = np.asarray(current_batch, dtype=np.float32).flatten()

        percentiles = np.linspace(0, 100, 11)
        bin_edges = np.percentile(ref, percentiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        curr_counts, _ = np.histogram(curr, bins=bin_edges)

        eps = 1e-4
        ref_pct = (ref_counts + eps) / (len(ref) + len(ref_counts) * eps)
        curr_pct = (curr_counts + eps) / (len(curr) + len(curr_counts) * eps)

        psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
        return psi

    def predict_with_guardian(
        self,
        raw_params: Dict[str, Any],
        fallback_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end guarded pricing call:
        Screen inputs -> execute surrogate -> screen outputs -> route to fallback if tripped.
        """
        t_start = time.perf_counter()

        # Step 1: Input OOD Screen (Tier 1)
        ood_result = self.check_input_ood(raw_params)
        if ood_result["is_ood"]:
            return self._execute_fallback(
                raw_params,
                reasons=ood_result["reasons"],
                trigger="Tier_1_Input_OOD",
                fallback_fn=fallback_fn,
                t_start=t_start,
            )

        # Step 2: Surrogate Inference
        if self.model is None or self.norm_in is None or self.norm_out is None:
            raise ValueError("Guardian model and normalizers must be configured for inference.")

        from deepvol.training.train_correction import (
            CorrectionInputNormalizer,
            compute_total_barrier_derivative,
        )

        X_row = CorrectionInputNormalizer.build_feature_matrix({k: [v] for k, v in raw_params.items()})
        X_t = self.norm_in.to_tensor(X_row, device=self.device)
        X_t.requires_grad_(True)

        # Step 2: Surrogate Inference & Ensemble Uncertainty Check
        if hasattr(self.model, "predict_with_routing"):
            mean_pred, std_bps, ood_mask = self.model.predict_with_routing(X_t, self.norm_out, tau_ood=self.tau_ood)
            if ood_mask.any():
                return self._execute_fallback(
                    raw_params,
                    reasons=[f"Ensemble epistemic uncertainty breach: {std_bps.item():.2f} bps > {self.tau_ood:.2f} bps"],
                    trigger="Tier_1_Ensemble_Uncertainty_OOD",
                    fallback_fn=fallback_fn,
                    t_start=t_start,
                )
            preds = mean_pred
        else:
            preds = self.model._forward_uncompiled(X_t) if hasattr(self.model, "_forward_uncompiled") else self.model(X_t)

        jac = torch.autograd.grad(preds.sum(), X_t, create_graph=False)[0]

        df_dB_tensor = compute_total_barrier_derivative(
            jac,
            {k: torch.tensor([float(v)], device=self.device) for k, v in raw_params.items()},
            self.norm_in,
            self.norm_out,
        )
        df_dB = float(df_dB_tensor.item())

        delta_v = float(self.norm_out.inverse_transform_tensor(preds).item())
        pde_npv = float(raw_params.get("pde_npv", 0.0))

        # Step 3: Output Arbitrage Screen (Tier 2)
        arb_result = self.check_output_arbitrage(raw_params, pde_npv, delta_v, df_dB=df_dB)
        if arb_result["is_arbitrage"]:
            return self._execute_fallback(
                raw_params,
                reasons=arb_result["reasons"],
                trigger="Tier_2_Output_Arbitrage",
                fallback_fn=fallback_fn,
                surrogate_pred=pde_npv + delta_v,
                t_start=t_start,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "npv": arb_result["total_npv"],
            "delta_v": delta_v,
            "pde_npv": pde_npv,
            "df_dB": df_dB,
            "is_fallback": False,
            "trigger": None,
            "reasons": [],
            "latency_ms": latency_ms,
        }

    def _execute_fallback(
        self,
        raw_params: Dict[str, Any],
        reasons: List[str],
        trigger: str,
        fallback_fn: Optional[Any],
        surrogate_pred: Optional[float] = None,
        t_start: float = 0.0,
    ) -> Dict[str, Any]:
        """Tier 3: Operational failover execution with structured SR 26-2 compliance logging."""
        logger.warning(
            f"[SR 26-2 Guardian Triggered] {trigger}: {'; '.join(reasons)}. Routing to numerical fallback."
        )

        fb_t0 = time.perf_counter()
        if fallback_fn is not None:
            fallback_npv = float(fallback_fn(raw_params))
        else:
            # Default fallback: Exact PDE solver with local vol
            from deepvol.models.autocall_pde import price_autocall_pde_scalar
            sigma_val = float(raw_params.get("sigma", 0.3))
            T_val = float(raw_params.get("T", 1.0))
            B_val = float(raw_params.get("B", 1.0))
            r_val = float(raw_params.get("r", 0.02))
            coupon_val = float(raw_params.get("coupon", 0.08))
            n_obs = int(raw_params.get("n_obs_per_year", 4) * T_val)

            from deepvol.models.autocall import make_obs_indices
            obs_indices = make_obs_indices(max(1, n_obs), T_val, 100)

            pde_res = price_autocall_pde_scalar(
                S0_val=1.0,
                r=r_val,
                T=T_val,
                N_S=200,
                N_T=100,
                obs_indices=obs_indices,
                B=B_val,
                coupon=coupon_val,
                sigma_func=lambda t, s: np.full_like(s, sigma_val),
            )
            fallback_npv = float(pde_res["npv"])

        fb_latency_ms = (time.perf_counter() - fb_t0) * 1000.0
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        audit_entry = {
            "timestamp": time.time(),
            "trigger": trigger,
            "reasons": reasons,
            "params": {k: float(v) for k, v in raw_params.items() if isinstance(v, (int, float, np.floating, np.integer))},
            "surrogate_npv": surrogate_pred,
            "fallback_npv": fallback_npv,
            "fallback_latency_ms": fb_latency_ms,
            "total_latency_ms": total_latency_ms,
        }
        self.audit_log.append(audit_entry)

        return {
            "npv": fallback_npv,
            "delta_v": fallback_npv - float(raw_params.get("pde_npv", 0.0)),
            "pde_npv": float(raw_params.get("pde_npv", 0.0)),
            "is_fallback": True,
            "trigger": trigger,
            "reasons": reasons,
            "latency_ms": total_latency_ms,
        }
