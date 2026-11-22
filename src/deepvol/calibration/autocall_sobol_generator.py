"""Vectorized Bartlett-LKJ sampler and GPU Dupire QMC dataset generator (v2.0)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc
import torch

from deepvol.models.autocall.analytical_baseline import (
    AutocallBaselineOutput,
    _digital_call_spread_prob_torch,
    compute_analytical_autocall_baseline,
)
from deepvol.models.autocall.dupire_qmc_gpu import DupireQMCConfig, DupireQMCGpuSolver
from deepvol.models.autocall.dupire_qmc_solver import RegularizedVolSurface
from deepvol.models.autocall.escrowed_dividend import EscrowedDividendOutput
from deepvol.models.autocall.feature_layout import (
    D_MAX_DEFAULT,
    FEATURE_SCHEMA_VERSION,
    M_MAX_DEFAULT,
    PAIR_ORDER_DEPENDENT_INDICES,
    AutocallNodeCol,
    monitoring_index_to_code,
)
from deepvol.models.autocall.pre_post_processor import AutocallPreprocState
from deepvol.models.autocall.seasoned_collapser import (
    SeasonedScheduleState,
    SentinelBarrierResult,
)

_SSVI_TENORS: npt.NDArray[np.float64] = np.array(
    [0.08, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 13.0], dtype=np.float64
)
_SSVI_STRIKES: npt.NDArray[np.float64] = np.array(
    [20.0, 35.0, 50.0, 65.0, 80.0, 90.0, 100.0, 110.0, 125.0, 145.0, 175.0, 220.0],
    dtype=np.float64,
)
_SSVI_LOG_K: npt.NDArray[np.float64] = np.log(_SSVI_STRIKES / 100.0)
_BASE_AS_OF: datetime = datetime(2026, 10, 23, 0, 0, 0, tzinfo=timezone.utc)

_CALL_FUNCTIONAL_CHOICES: Tuple[str, ...] = (
    "WorstOf",
    "BestOf",
    "AverageOf",
    "Dispersion",
    "RatioAB",
    "RatioBA",
    "DiffAB",
    "DiffBA",
)


@dataclass
class SyntheticAutocallBatch:
    """Batch of v2.0 autocallable features, analytical baselines, Dupire QMC targets, and Jacobians."""

    # Padded Tier 2 input tensors
    node_feats: torch.Tensor    # (B, D_max, 16)
    edge_feats: torch.Tensor    # (B, D_max, D_max, 4)
    asset_mask: torch.Tensor    # (B, D_max)
    sched_feats: torch.Tensor   # (B, M_max, 12)
    sched_mask: torch.Tensor    # (B, M_max)
    global_feats: torch.Tensor  # (B, 16)

    # Analytical Stulz-Margrabe proxy baseline
    p_proxy: torch.Tensor       # (B, M_max)
    q_proxy: torch.Tensor       # (B, M_max)
    l_put_proxy: torch.Tensor   # (B,)
    pv_proxy: torch.Tensor      # (B,)
    delta_proxy: torch.Tensor   # (B, D_max) dPV_proxy / d(ln M_{0,i})

    # Multi-asset GPU Dupire LV QMC target labels
    p_mc: torch.Tensor          # (B, M_max)
    q_mc: torch.Tensor          # (B, M_max)
    l_put_mc: torch.Tensor      # (B,)
    pv_mc: torch.Tensor         # (B,)
    delta_mc: torch.Tensor      # (B, D_max) unblended dPV_mc / d(ln M_{0,i})

    # Contract payoff modifiers & precomputed training tensors
    no_notional_repayment: torch.Tensor     # (B,)
    autocall_coupon: torch.Tensor           # (B,)
    force_coupons_on_autocall: torch.Tensor # (B,)
    is_long_basket: torch.Tensor            # (B,)
    lower_strike: torch.Tensor              # (B,)
    p_proxy_ko_bumped: torch.Tensor         # (B, M_max)
    jac_node_logm: torch.Tensor             # (B, D_max, D_max, 16)
    jac_edge_self: torch.Tensor             # (B, D_max, D_max, 2) for cols [COVARIANCE_TERM, LOG_PERF_DIFF]
    jac_sched_logm: torch.Tensor            # (B, D_max, M_max, 4) for cols [8..11]
    jac_lput_logm: torch.Tensor             # (B, D_max)

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> SyntheticAutocallBatch:
        """Move all tensors in the batch to `device` and/or floating `dtype`."""
        def _mv(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device if device is not None else t.device, dtype=dtype if dtype is not None else t.dtype)

        return SyntheticAutocallBatch(
            node_feats=_mv(self.node_feats),
            edge_feats=_mv(self.edge_feats),
            asset_mask=_mv(self.asset_mask),
            sched_feats=_mv(self.sched_feats),
            sched_mask=_mv(self.sched_mask),
            global_feats=_mv(self.global_feats),
            p_proxy=_mv(self.p_proxy),
            q_proxy=_mv(self.q_proxy),
            l_put_proxy=_mv(self.l_put_proxy),
            pv_proxy=_mv(self.pv_proxy),
            delta_proxy=_mv(self.delta_proxy),
            p_mc=_mv(self.p_mc),
            q_mc=_mv(self.q_mc),
            l_put_mc=_mv(self.l_put_mc),
            pv_mc=_mv(self.pv_mc),
            delta_mc=_mv(self.delta_mc),
            no_notional_repayment=_mv(self.no_notional_repayment),
            autocall_coupon=_mv(self.autocall_coupon),
            force_coupons_on_autocall=_mv(self.force_coupons_on_autocall),
            is_long_basket=_mv(self.is_long_basket),
            lower_strike=_mv(self.lower_strike),
            p_proxy_ko_bumped=_mv(self.p_proxy_ko_bumped),
            jac_node_logm=_mv(self.jac_node_logm),
            jac_edge_self=_mv(self.jac_edge_self),
            jac_sched_logm=_mv(self.jac_sched_logm),
            jac_lput_logm=_mv(self.jac_lput_logm),
        )

    def index_select(self, idx: torch.Tensor) -> SyntheticAutocallBatch:
        """Gather a mini-batch along dimension 0."""
        idx_dev = idx.to(device=self.node_feats.device)
        return SyntheticAutocallBatch(
            node_feats=self.node_feats.index_select(0, idx_dev),
            edge_feats=self.edge_feats.index_select(0, idx_dev),
            asset_mask=self.asset_mask.index_select(0, idx_dev),
            sched_feats=self.sched_feats.index_select(0, idx_dev),
            sched_mask=self.sched_mask.index_select(0, idx_dev),
            global_feats=self.global_feats.index_select(0, idx_dev),
            p_proxy=self.p_proxy.index_select(0, idx_dev),
            q_proxy=self.q_proxy.index_select(0, idx_dev),
            l_put_proxy=self.l_put_proxy.index_select(0, idx_dev),
            pv_proxy=self.pv_proxy.index_select(0, idx_dev),
            delta_proxy=self.delta_proxy.index_select(0, idx_dev),
            p_mc=self.p_mc.index_select(0, idx_dev),
            q_mc=self.q_mc.index_select(0, idx_dev),
            l_put_mc=self.l_put_mc.index_select(0, idx_dev),
            pv_mc=self.pv_mc.index_select(0, idx_dev),
            delta_mc=self.delta_mc.index_select(0, idx_dev),
            no_notional_repayment=self.no_notional_repayment.index_select(0, idx_dev),
            autocall_coupon=self.autocall_coupon.index_select(0, idx_dev),
            force_coupons_on_autocall=self.force_coupons_on_autocall.index_select(0, idx_dev),
            is_long_basket=self.is_long_basket.index_select(0, idx_dev),
            lower_strike=self.lower_strike.index_select(0, idx_dev),
            p_proxy_ko_bumped=self.p_proxy_ko_bumped.index_select(0, idx_dev),
            jac_node_logm=self.jac_node_logm.index_select(0, idx_dev),
            jac_edge_self=self.jac_edge_self.index_select(0, idx_dev),
            jac_sched_logm=self.jac_sched_logm.index_select(0, idx_dev),
            jac_lput_logm=self.jac_lput_logm.index_select(0, idx_dev),
        )


def _sample_symmetric_beta(
    alpha: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Samples Beta(alpha, alpha) via independent Gamma(alpha, 1) variates."""
    g1 = torch._standard_gamma(alpha, generator=generator).clamp(min=1e-30)
    g2 = torch._standard_gamma(alpha, generator=generator).clamp(min=1e-30)
    return (g1 / (g1 + g2)).clamp(1e-7, 1.0 - 1e-7)


def sample_lkj_cholesky_bartlett(
    batch_size: int,
    d: int,
    eta: float = 2.0,
    crisis_prob: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples random SPD correlation matrices R and lower Cholesky factors L from LKJ(eta)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if d <= 0:
        raise ValueError(f"dimension d must be >= 1, got {d}")
    if eta <= 0.0:
        raise ValueError(f"LKJ shape parameter eta must be > 0, got {eta}")
    if not (0.0 <= crisis_prob <= 1.0):
        raise ValueError(f"crisis_prob must be in [0, 1], got {crisis_prob}")

    dev = torch.device(device)
    work_dtype = torch.float64

    if d == 1:
        ones = torch.ones((batch_size, 1, 1), dtype=dtype, device=dev)
        return ones, ones.clone()

    chol = torch.zeros((batch_size, d, d), dtype=work_dtype, device=dev)
    chol[:, 0, 0] = 1.0

    for j in range(1, d):
        k_idx = torch.arange(j, dtype=work_dtype, device=dev)
        alphas = float(eta) + 0.5 * (float(d - 2) - k_idx)
        alpha_batch = alphas.unsqueeze(0).expand(batch_size, j).contiguous()
        u_kj = _sample_symmetric_beta(alpha_batch, generator=generator)
        p_kj = (2.0 * u_kj - 1.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sin_kj = torch.sqrt((1.0 - p_kj * p_kj).clamp(min=1e-14))
        cum_sin = torch.cumprod(sin_kj, dim=1)
        prefix_sin = torch.cat(
            [torch.ones((batch_size, 1), dtype=work_dtype, device=dev), cum_sin[:, :-1]],
            dim=1,
        )
        chol[:, j, :j] = p_kj * prefix_sin
        chol[:, j, j] = cum_sin[:, -1]

    if crisis_prob > 0.0:
        u_crisis = torch.rand((batch_size,), dtype=work_dtype, device=dev, generator=generator)
        is_crisis = u_crisis < float(crisis_prob)
        n_crisis = int(is_crisis.sum().item())
        if n_crisis > 0:
            rho_c = 0.80 + 0.15 * torch.rand(
                (n_crisis, 1, 1), dtype=work_dtype, device=dev, generator=generator
            )
            eye_d = torch.eye(d, dtype=work_dtype, device=dev).unsqueeze(0)
            r_crisis = rho_c * (1.0 - eye_d) + eye_d
            l_crisis = torch.linalg.cholesky(r_crisis)
            chol[is_crisis] = l_crisis

    corr = torch.bmm(chol, chol.transpose(1, 2))
    corr = 0.5 * (corr + corr.transpose(1, 2))
    corr.diagonal(dim1=-2, dim2=-1).fill_(1.0)

    return corr.to(dtype=dtype), chol.to(dtype=dtype)


def _build_ssvi_surface(
    identifier: str,
    iv_short: float,
    iv_slope: float,
    rho: float,
    eta: float,
) -> RegularizedVolSurface:
    """Builds an arbitrage-free Gatheral SSVI RegularizedVolSurface in float64."""
    iv_atm = np.clip(
        iv_short + iv_slope * np.minimum(_SSVI_TENORS / 2.0, 1.0),
        0.08,
        0.90,
    )
    theta = (iv_atm * iv_atm) * _SSVI_TENORS
    for m in range(1, len(theta)):
        min_t = theta[m - 1] + 5e-4 * (_SSVI_TENORS[m] - _SSVI_TENORS[m - 1])
        if theta[m] < min_t:
            theta[m] = min_t

    abs_rho = abs(rho)
    # Enforce Gatheral-Jacquier butterfly sufficiency: theta*phi*(1+|rho|) < 4 and theta*phi^2*(1+|rho|) <= 4.
    phi_raw = eta / np.sqrt(np.maximum(theta, 0.04))
    phi_cap1 = 1.8 / (np.maximum(theta, 1e-4) * (1.0 + abs_rho))
    phi_cap2 = np.sqrt(3.2 / (np.maximum(theta, 1e-4) * (1.0 + abs_rho)))
    phi = np.minimum(phi_raw, np.minimum(phi_cap1, phi_cap2))

    k_2d = _SSVI_LOG_K[None, :]
    phi_2d = phi[:, None]
    theta_2d = theta[:, None]
    pk = phi_2d * k_2d
    w_grid = 0.5 * theta_2d * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + max(1.0 - rho * rho, 1e-4)))
    quotes = np.clip(np.sqrt(np.maximum(w_grid / _SSVI_TENORS[:, None], 1e-4)), 0.05, 2.50)

    spec = {
        "Identifier": identifier,
        "tenors": _SSVI_TENORS,
        "Strikes": _SSVI_STRIKES,
        "VolatilityQuotes": quotes,
        "DayCountConvention": "Act365",
        "SmileInterpolationMethod": "CubicSpline",
        "SmileExtrapolationMethod": "Flat",
    }
    return RegularizedVolSurface(spec, min_vol=0.01, max_local_vol=2.50)


_SSVI_BANK_CACHE: Dict[Tuple[int, str], List[RegularizedVolSurface]] = {}


def _get_ssvi_bank(
    n_bank: int = 1024,
    *,
    domain_mode: str = "full_stress",
) -> List[RegularizedVolSurface]:
    """Lazily builds and caches a per-process Sobol-stratified SSVI surface bank."""
    key = (int(n_bank), str(domain_mode))
    cached = _SSVI_BANK_CACHE.get(key)
    if cached is not None:
        return cached
    eng = qmc.Sobol(d=4, scramble=True, seed=10007 if domain_mode == "full_stress" else 10009)
    n_pow = 1 << max(1, (n_bank - 1).bit_length())
    u_bank = eng.random(n=n_pow)[:n_bank]
    bank: List[RegularizedVolSurface] = []
    for k in range(n_bank):
        if domain_mode == "core_desk":
            # Institutional desk manifold: iv_atm in [0.15, 0.45] with realistic equity skew/curvature.
            iv_s = 0.15 + 0.27 * float(u_bank[k, 0])
            iv_sl = -0.04 + 0.09 * float(u_bank[k, 1])
            rho_k = -0.55 + 0.65 * float(u_bank[k, 2])
            eta_k = 0.15 + 0.30 * float(u_bank[k, 3])
        else:
            iv_s = 0.14 + 0.41 * float(u_bank[k, 0])
            iv_sl = -0.07 + 0.16 * float(u_bank[k, 1])
            rho_k = -0.65 + 0.90 * float(u_bank[k, 2])
            eta_k = 0.15 + 0.55 * float(u_bank[k, 3])
        bank.append(_build_ssvi_surface(f"BANK_{domain_mode}_{k:04d}", iv_s, iv_sl, rho_k, eta_k))
    _SSVI_BANK_CACHE[key] = bank
    return bank


def _sample_owen_sobol_matrix(n_samples: int, dim: int, seed: int) -> npt.NDArray[np.float64]:
    """Draws (n_samples, dim) independent Owen-scrambled Sobol uniforms in (0, 1)."""
    eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
    n_pow = 1 << max(1, (n_samples - 1).bit_length())
    draws = eng.random(n=n_pow)[:n_samples]
    return np.asarray(draws, dtype=np.float64)


def _assemble_w1_pv_torch(
    baseline: AutocallBaselineOutput,
    df_pay: torch.Tensor,
    cpn_rates: torch.Tensor,
    *,
    strike: float,
    option_gearing: float,
    c_mem_0: float,
    is_memory: bool,
    no_notional_repayment: bool,
    autocall_coupon: float,
    force_coupons_on_autocall: bool,
    is_long_basket: bool,
) -> torch.Tensor:
    """Pure differentiable W1 ProductValue matching assemble_surrogate_product_value."""
    m_rem = int(df_pay.shape[0])
    p_m = baseline.p_proxy
    q_m = baseline.q_proxy
    surv_m = baseline.survival_proxy
    surv_prev = torch.cat([torch.ones(1, dtype=torch.float64, device=df_pay.device), surv_m[:-1]], dim=0)

    if no_notional_repayment:
        pv_notional = torch.zeros((), dtype=torch.float64, device=df_pay.device)
    else:
        pv_notional = (df_pay * p_m).sum() + df_pay[-1] * surv_m[-1]

    eff_q_all = (
        torch.maximum(q_m, p_m / surv_prev.clamp(min=1e-12))
        if force_coupons_on_autocall
        else q_m
    )
    if not is_memory:
        pv_cpn = (df_pay * (surv_prev * eff_q_all * cpn_rates + p_m * autocall_coupon)).sum()
    else:
        eq_u = eff_q_all.unbind(0)
        sp_u = surv_prev.unbind(0)
        df_u = df_pay.unbind(0)
        cp_u = cpn_rates.unbind(0)
        pm_u = p_m.unbind(0)
        mem_buf = torch.tensor(c_mem_0, dtype=torch.float64, device=df_pay.device)
        pv_cpn = torch.zeros((), dtype=torch.float64, device=df_pay.device)
        for m in range(m_rem):
            c_eff = cp_u[m] + mem_buf
            pv_cpn = pv_cpn + df_u[m] * (sp_u[m] * eq_u[m] * c_eff + pm_u[m] * autocall_coupon)
            mem_buf = (1.0 - eq_u[m]) * c_eff

    _ = is_long_basket
    put_sign = -1.0
    pv_opt = put_sign * df_pay[-1] * option_gearing * baseline.l_put_proxy
    return pv_notional + pv_cpn + pv_opt


def _build_contract_state_and_jacs(
    *,
    u_row: npt.NDArray[np.float64],
    d_active: int,
    m_rem: int,
    d_max: int,
    m_max: int,
    ko_bump_delta: float,
    rng: np.random.Generator,
    attach_surfaces: bool = True,
    domain_mode: str = "tier2_desk",
) -> Tuple[
    AutocallPreprocState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
]:
    """Builds one contract's AutocallPreprocState, exact dX/d(ln M0) Jacobians, and bumped KO probs."""
    u_reg = float(u_row[0])
    m0_u = rng.random(d_active)
    is_distressed = False
    if domain_mode == "tier2_desk":
        is_distressed = u_reg >= 0.85
        if not is_distressed:
            # 85% core desk manifold: m0 in [0.60, 1.35]
            m0_np = 0.60 + 0.75 * m0_u
        else:
            # 15% seasoned/distressed belt (especially M_rem <= 4 so reference 45476647 is well in-support)
            if m_rem <= 4 and float(rng.random()) < 0.70:
                m0_np = 0.18 + 0.55 * m0_u
            else:
                m0_np = 0.18 + 1.22 * m0_u
    else:
        if u_reg < 0.25:
            m0_np = 0.15 + 0.50 * m0_u
        elif u_reg < 0.60:
            m0_np = 0.75 + 0.40 * m0_u
        else:
            m0_np = 0.35 + 1.15 * m0_u
    # Exercise exact/near-exact ties on ~10% of multi-asset contracts.
    if d_active >= 2 and float(u_row[1]) < 0.10:
        m0_np[1] = m0_np[0]

    ref_np = np.full(d_active, 100.0, dtype=np.float64)
    spots_np = ref_np * m0_np

    r_disc = (0.02 + 0.20 * float(u_row[2])) if domain_mode == "tier2_desk" else (0.02 + 0.14 * float(u_row[2]))
    div_yields = 0.04 * rng.random(d_active)
    repo_avg_np = 0.03 * rng.random(d_active)
    carry_1y_np = r_disc - div_yields - repo_avg_np

    u_dtau = float(u_row[3])
    if domain_mode == "tier2_desk":
        # Cap contract maturity <= 5.0y across M_rem in [1..48]
        if m_rem > 20:
            dtau = 1.0 / 12.0
        elif m_rem >= 9:
            dtau = (1.0 / 12.0) if u_dtau < 0.50 else 0.25
        else:
            dtau = 0.15 + 0.35 * u_dtau
    else:
        if m_rem <= 12:
            dtau = 0.08333333333333333 if u_dtau < 0.25 else (0.25 if u_dtau < 0.80 else 0.50)
        elif m_rem <= 24:
            dtau = 0.08333333333333333 if u_dtau < 0.35 else (0.25 if u_dtau < 0.90 else 0.50)
        else:
            dtau = 0.08333333333333333 if u_dtau < 0.45 else 0.25

    tau_obs_np = np.arange(1, m_rem + 1, dtype=np.float64) * dtau
    delta_tau_np = np.full(m_rem, dtau, dtype=np.float64)
    df_pay_np = np.exp(-r_disc * tau_obs_np)

    u_fn = float(u_row[4])
    if domain_mode == "tier2_desk":
        if d_active < 2 or u_fn < 0.60:
            call_idx = "WorstOf"
        elif u_fn < 0.80:
            call_idx = "BestOf"
        else:
            call_idx = "AverageOf"
        cpn_idx = call_idx
        opt_idx = call_idx
    elif d_active < 2:
        call_idx = "WorstOf"
        cpn_idx = "WorstOf"
        opt_idx = "WorstOf"
    elif u_fn < 0.50:
        call_idx = "WorstOf"
        cpn_idx = "WorstOf"
        opt_idx = "WorstOf"
    elif u_fn < 0.62:
        call_idx = "BestOf"
        cpn_idx = "BestOf"
        opt_idx = "BestOf"
    elif u_fn < 0.74:
        call_idx = "AverageOf"
        cpn_idx = "AverageOf"
        opt_idx = "AverageOf"
    elif u_fn < 0.82:
        call_idx = "Dispersion"
        cpn_idx = "Dispersion"
        opt_idx = "WorstOf" if float(u_row[5]) < 0.5 else "Dispersion"
    elif u_fn < 0.88 and d_active == 2:
        call_idx = "RatioAB" if float(u_row[5]) < 0.5 else "RatioBA"
        cpn_idx = call_idx
        opt_idx = call_idx
    elif u_fn < 0.94 and d_active == 2:
        call_idx = "DiffAB" if float(u_row[5]) < 0.5 else "DiffBA"
        cpn_idx = call_idx
        # Signed differences can go negative and exceed K_put; WorstOf put keeps the upper cap U_put exact.
        opt_idx = "WorstOf"
    else:
        pool_3 = ("WorstOf", "BestOf", "AverageOf")
        call_idx = pool_3[int(rng.integers(0, 3))]
        cpn_idx = pool_3[int(rng.integers(0, 3))]
        opt_idx = pool_3[int(rng.integers(0, 3))]

    sorted_ids = [f"SYN_{i:02d}" for i in range(d_active)]
    lex_perm_np = np.arange(d_active, dtype=np.int64)
    if d_active >= 2 and float(u_row[6]) < 0.5:
        inv_lex_perm_np = np.arange(d_active, dtype=np.int64)
        inv_lex_perm_np[0], inv_lex_perm_np[1] = 1, 0
    else:
        inv_lex_perm_np = np.arange(d_active, dtype=np.int64)

    role_leg_a = np.zeros(d_active, dtype=np.float64)
    role_leg_b = np.zeros(d_active, dtype=np.float64)
    if d_active >= 2 and any(idx in PAIR_ORDER_DEPENDENT_INDICES for idx in (call_idx, cpn_idx, opt_idx)):
        role_leg_a[int(inv_lex_perm_np[0])] = 1.0
        role_leg_b[int(inv_lex_perm_np[1])] = 1.0

    # LKJ correlation matrix with crisis regime mixing
    gen_corr = torch.Generator(device="cpu")
    gen_corr.manual_seed(int(rng.integers(0, 2**31 - 1)))
    corr_b, _ = sample_lkj_cholesky_bartlett(
        batch_size=1,
        d=d_active,
        eta=2.0,
        crisis_prob=0.15,
        device="cpu",
        dtype=torch.float64,
        generator=gen_corr,
    )
    corr_t = corr_b[0]
    if domain_mode == "tier2_desk" and d_active >= 2:
        rho_pos = 0.15 + 0.60 * float(u_row[5])
        eye_c = torch.eye(d_active, dtype=torch.float64)
        r_pos = rho_pos * (1.0 - eye_c) + eye_c
        corr_t = 0.35 * corr_t + 0.65 * r_pos
        corr_t.diagonal().fill_(1.0)

    # Contract payoff terms & independent digital spreads
    if opt_idx == "Dispersion":
        k_put = 0.05 + 0.20 * float(u_row[7])
    else:
        k_put = 0.65 + 0.35 * float(u_row[7])
    has_lower = float(u_row[8]) < 0.25 and k_put > 0.12
    if domain_mode == "tier2_desk":
        lower_strike_val = (
            round(min(0.35, k_put * 0.60) * float(rng.random()), 4) if has_lower else 0.0
        )
    else:
        lower_strike_val = round(k_put * (0.25 + 0.45 * float(rng.random())), 4) if has_lower else 0.0
    has_ki = float(u_row[9]) < 0.65
    if opt_idx == "Dispersion":
        b_ki = (0.03 + 0.12 * float(rng.random())) if has_ki else k_put
    else:
        b_ki = (0.45 + 0.35 * float(rng.random())) if has_ki else k_put
    i_ki_past = 1.0 if (has_ki and float(u_row[10]) < 0.10) else 0.0
    if domain_mode == "tier2_desk":
        g_put = (
            min(1.25, max(0.50, 1.0 / max(k_put, 1e-4)))
            if float(u_row[11]) < 0.50
            else (0.50 + 0.75 * float(rng.random()))
        )
    else:
        g_put = (
            min(2.0, 1.0 / max(k_put, 1e-4))
            if float(u_row[11]) < 0.50
            else (0.5 + 1.2 * float(rng.random()))
        )
    is_memory = float(u_row[12]) < 0.65
    c_mem_cap = 0.40 if (domain_mode == "tier2_desk" and is_distressed) else (0.38 if domain_mode == "tier2_desk" else 0.45)
    c_mem_0 = (float(rng.random()) * c_mem_cap) if (is_memory and float(rng.random()) < 0.70) else 0.0

    no_notional = float(u_row[13]) < 0.08
    ac_cpn = (0.02 + 0.10 * float(rng.random())) if float(u_row[14]) < 0.12 else 0.0
    force_cpn = float(u_row[15]) < 0.18
    is_long_basket = False if domain_mode == "tier2_desk" else (float(u_row[16]) < 0.10)

    def _draw_spread() -> float:
        if domain_mode == "tier2_desk":
            u_sp = float(rng.random())
            return 0.0 if u_sp < 0.03 else -(0.005 + 0.145 * ((u_sp - 0.03) / 0.97))
        u_s = float(rng.random())
        if u_s < 0.05:
            return 0.0
        return -(0.005 + 0.195 * float(rng.random()))

    ko_spread = _draw_spread()
    cpn_spread = _draw_spread()
    ki_spread = _draw_spread()

    # Scale barriers appropriately for Dispersion / Diff* / Ratio* vs standard level functionals.
    def _barrier_scale(idx_name: str, is_ko: bool) -> Tuple[float, float]:
        if idx_name == "Dispersion":
            return (0.18 + 0.25 * float(rng.random()), 0.005) if is_ko else (0.08 + 0.15 * float(rng.random()), 0.0)
        if idx_name in ("DiffAB", "DiffBA"):
            return (0.10 + 0.25 * float(rng.random()), 0.005) if is_ko else (-0.15 + 0.20 * float(rng.random()), 0.0)
        if idx_name in ("RatioAB", "RatioBA"):
            return (1.05 + 0.20 * float(rng.random()), 0.01) if is_ko else (0.75 + 0.20 * float(rng.random()), 0.0)
        return (0.95 + 0.15 * float(rng.random()), 0.015 * float(rng.random())) if is_ko else (0.65 + 0.20 * float(rng.random()), 0.0)

    base_ko, step_ko = _barrier_scale(call_idx, True)
    base_cpn_b, _ = _barrier_scale(cpn_idx, False)
    if domain_mode == "tier2_desk":
        # FixedRate * dtau per Murex schedule convention (6%-20% annualized coupon)
        cpn_rate_val = (0.06 + 0.14 * float(rng.random())) * dtau
    else:
        cpn_rate_val = 0.02 + 0.03 * float(rng.random())

    ko_barriers_np = np.zeros(m_rem, dtype=np.float64)
    is_callable_np = np.ones(m_rem, dtype=np.float64)
    cpn_barriers_np = np.zeros(m_rem, dtype=np.float64)
    is_guar_np = np.zeros(m_rem, dtype=np.float64)
    cpn_rates_np = np.full(m_rem, cpn_rate_val, dtype=np.float64)
    log_ko_feat_np = np.zeros(m_rem, dtype=np.float64)
    log_cpn_feat_np = np.zeros(m_rem, dtype=np.float64)

    lockout_len = int(rng.integers(1, min(4, m_rem))) if (m_rem > 1 and float(u_row[17]) < 0.25) else 0
    has_guar_0 = float(u_row[18]) < 0.20
    for m in range(m_rem):
        # allow_autocall_on_expiry=False requires final expiry date m == m_rem - 1 to be non-callable
        # so Tier-1 proxy does not double-discount L_put by (1 - p[-1]).
        if m < lockout_len or m == (m_rem - 1):
            ko_barriers_np[m] = 500.0
            is_callable_np[m] = 0.0
            log_ko_feat_np[m] = math.log(3.0)
        else:
            b_k = max(base_ko - step_ko * m, 0.50 if call_idx not in ("Dispersion", "DiffAB", "DiffBA") else 0.05)
            ko_barriers_np[m] = b_k
            is_callable_np[m] = 1.0
            log_ko_feat_np[m] = math.log(min(max(b_k, 1e-4), 3.0))

        if m == 0 and has_guar_0:
            cpn_barriers_np[m] = 1e-5
            is_guar_np[m] = 1.0
            log_cpn_feat_np[m] = math.log(0.05)
        else:
            cpn_barriers_np[m] = base_cpn_b
            is_guar_np[m] = 0.0
            log_cpn_feat_np[m] = math.log(max(base_cpn_b, 0.05))

    # Draw arbitrage-free SSVI RegularizedVolSurfaces from cached bank and precompute linearizations
    bank_mode = "full_stress" if (domain_mode != "tier2_desk" or is_distressed) else "core_desk"
    ssvi_bank = _get_ssvi_bank(1024, domain_mode=bank_mode)
    surf_indices: List[int] = [int(idx) for idx in rng.integers(0, len(ssvi_bank), size=d_active)]
    reg_surfaces: Dict[str, RegularizedVolSurface] = {}
    iv_0_np = np.zeros((d_active, m_rem), dtype=np.float64)
    d_iv_dk_np = np.zeros((d_active, m_rem), dtype=np.float64)
    surf_feats_0_np = np.zeros((d_active, 7), dtype=np.float64)
    d_surf_df_short_np = np.zeros((d_active, 7), dtype=np.float64)
    d_surf_df_term_np = np.zeros((d_active, 7), dtype=np.float64)

    fwd_rel_np = np.exp(carry_1y_np[:, None] * tau_obs_np[None, :])
    fwd_0_np = spots_np[:, None] * fwd_rel_np

    for i, ticker in enumerate(sorted_ids):
        surf_i = ssvi_bank[surf_indices[i]]
        if attach_surfaces:
            reg_surfaces[ticker] = surf_i
        f_row = fwd_0_np[i]
        iv_0_np[i] = surf_i.iv_batch(tau_obs_np, f_row)
        eps_k = np.maximum(1e-5 * f_row, 1e-6)
        f_hi = f_row + eps_k
        f_lo = np.maximum(f_row - eps_k, 1e-6)
        d_iv_dk_np[i] = (surf_i.iv_batch(tau_obs_np, f_hi) - surf_i.iv_batch(tau_obs_np, f_lo)) / (f_hi - f_lo)

        t_s = float(tau_obs_np[0])
        t_t = float(tau_obs_np[-1])
        f_s = float(f_row[0])
        f_t = float(f_row[-1])
        surf_feats_0_np[i] = surf_i.extract_node_surface_features_values_only(
            t_s, f_s, t_t, f_t, iv_short_val=float(iv_0_np[i, 0]), iv_term_val=float(iv_0_np[i, -1])
        )
        es = max(1e-4 * f_s, 1e-5)
        et = max(1e-4 * f_t, 1e-5)
        d_surf_df_short_np[i] = (
            surf_i.extract_node_surface_features_values_only(t_s, f_s + es, t_t, f_t)
            - surf_i.extract_node_surface_features_values_only(t_s, max(f_s - es, 1e-5), t_t, f_t)
        ) / (f_s + es - max(f_s - es, 1e-5))
        d_surf_df_term_np[i] = (
            surf_i.extract_node_surface_features_values_only(t_s, f_s, t_t, f_t + et)
            - surf_i.extract_node_surface_features_values_only(t_s, f_s, t_t, max(f_t - et, 1e-5))
        ) / (f_t + et - max(f_t - et, 1e-5))

    # Tensors used inside jacfwd feature builder
    ref_t = torch.tensor(ref_np, dtype=torch.float64)
    fwd_rel_t = torch.tensor(fwd_rel_np, dtype=torch.float64)
    fwd_0_t = torch.tensor(fwd_0_np, dtype=torch.float64)
    iv_0_t = torch.tensor(iv_0_np, dtype=torch.float64)
    d_iv_dk_t = torch.tensor(d_iv_dk_np, dtype=torch.float64)
    surf_0_t = torch.tensor(surf_feats_0_np, dtype=torch.float64)
    d_surf_fs_t = torch.tensor(d_surf_df_short_np, dtype=torch.float64)
    d_surf_ft_t = torch.tensor(d_surf_df_term_np, dtype=torch.float64)
    sqrt_tau_t = torch.sqrt(torch.tensor(tau_obs_np, dtype=torch.float64).clamp(min=1e-12))

    carry_t = torch.tensor(carry_1y_np, dtype=torch.float64)
    repo_t = torch.tensor(repo_avg_np, dtype=torch.float64)
    div_lev_cap = (0.30 if float(u_row[16]) < 0.25 else 0.06) if domain_mode == "tier2_desk" else 0.06
    div_lev_np = np.where(rng.random(d_active) < 0.35, 0.0, div_lev_cap * rng.random(d_active))
    div_lev_log_t = torch.tensor(div_lev_np, dtype=torch.float64)
    cash_lev_t = torch.exp(div_lev_log_t).unsqueeze(1).expand(d_active, m_rem).clone()
    role_a_t = torch.tensor(role_leg_a, dtype=torch.float64)
    role_b_t = torch.tensor(role_leg_b, dtype=torch.float64)
    ones_d = torch.ones(d_active, dtype=torch.float64)

    tau_obs_t = torch.tensor(tau_obs_np, dtype=torch.float64)
    dtau_t = torch.tensor(delta_tau_np, dtype=torch.float64)
    df_pay_t = torch.tensor(df_pay_np, dtype=torch.float64)
    ko_t = torch.tensor(ko_barriers_np, dtype=torch.float64)
    is_call_t = torch.tensor(is_callable_np, dtype=torch.float64)
    cpn_b_t = torch.tensor(cpn_barriers_np, dtype=torch.float64)
    is_guar_t = torch.tensor(is_guar_np, dtype=torch.float64)
    cpn_r_t = torch.tensor(cpn_rates_np, dtype=torch.float64)
    log_ko_t = torch.tensor(log_ko_feat_np, dtype=torch.float64)
    log_cpn_t = torch.tensor(log_cpn_feat_np, dtype=torch.float64)

    call_code_f = float(monitoring_index_to_code(call_idx))
    cpn_code_f = float(monitoring_index_to_code(cpn_idx))
    opt_code_f = float(monitoring_index_to_code(opt_idx))
    eye_d = torch.eye(d_active, dtype=torch.float64)
    off_diag = 1.0 - eye_d
    k_eff_put = max(k_put, 1e-4)
    glob_base = torch.tensor(
        [
            k_put,
            b_ki,
            1.0 if has_ki else 0.0,
            i_ki_past,
            g_put,
            c_mem_0,
            1.0 if is_memory else 0.0,
            abs(ko_spread),
            abs(cpn_spread),
            0.0,
            lower_strike_val,
            call_code_f,
            cpn_code_f,
            opt_code_f,
            1.0 if is_long_basket else 0.0,
            abs(ki_spread),
        ],
        dtype=torch.float64,
    )

    def _features_from_m0(
        m0_vec: torch.Tensor,
        *,
        use_fast_cpu: Optional[bool] = True,
    ) -> Tuple[
        torch.Tensor,  # node_act (D, 16)
        torch.Tensor,  # edge_act (D, D, 4)
        torch.Tensor,  # sched_act (M, 12)
        torch.Tensor,  # glob_act (16,)
        torch.Tensor,  # pv_proxy ()
        torch.Tensor,  # p_proxy (M,)
        torch.Tensor,  # q_proxy (M,)
        torch.Tensor,  # l_put_proxy ()
        torch.Tensor,  # f_w (M,)
        torch.Tensor,  # v_w (M,)
    ]:
        spots_v = ref_t * m0_vec
        fwd_price_v = spots_v.unsqueeze(1) * fwd_rel_t
        fwd_perf_v = m0_vec.unsqueeze(1) * fwd_rel_t
        iv_v = (iv_0_t + d_iv_dk_t * (fwd_price_v - fwd_0_t)).clamp(min=0.01)
        v_perf_v = iv_v * sqrt_tau_t.unsqueeze(0) * cash_lev_t

        df_s = (fwd_price_v[:, 0] - fwd_0_t[:, 0]).unsqueeze(-1)
        df_t = (fwd_price_v[:, -1] - fwd_0_t[:, -1]).unsqueeze(-1)
        surf_v = surf_0_t + d_surf_fs_t * df_s + d_surf_ft_t * df_t

        proxy_in = {
            "forward_perf": fwd_perf_v,
            "v_perf": v_perf_v,
            "corr_matrix": corr_t,
            "df_pay": df_pay_t,
            "ko_barriers": ko_t,
            "is_callable": is_call_t,
            "cpn_barriers": cpn_b_t,
            "is_guaranteed_cpn": is_guar_t,
            "coupon_amounts": cpn_r_t,
            "strike": k_put,
            "option_gearing": g_put,
            "lower_strike": lower_strike_val if lower_strike_val > 0.0 else None,
            "ki_barrier": b_ki if has_ki else None,
            "i_ki_past": i_ki_past,
            "c_mem_0": c_mem_0,
            "is_memory": is_memory,
            "ko_spread": ko_spread,
            "cpn_spread": cpn_spread,
            "ki_spread": ki_spread,
            "no_notional_repayment": no_notional,
            "autocall_coupon": ac_cpn,
            "force_coupons_on_autocall": force_cpn,
            "allow_autocall_on_expiry": False,
            "is_long_basket": is_long_basket,
            "pv_in_transit": 0.0,
            "already_autocalled": False,
            "callability_index": call_idx,
            "coupon_index": cpn_idx,
            "option_index": opt_idx,
            "inv_lex_perm": inv_lex_perm_np,
            "tau_obs": tau_obs_t,
        }
        b_out = compute_analytical_autocall_baseline(proxy_in, use_fast_cpu=use_fast_cpu)

        log_m0_v = torch.log(m0_vec.clamp(min=1e-12))
        f_p_term = fwd_perf_v[:, -1].clamp(min=1e-12)
        v_term_v = v_perf_v[:, -1].clamp(min=1e-6)
        dist_put = torch.log(f_p_term / k_eff_put) / v_term_v

        node_v = torch.stack(
            [
                log_m0_v,
                carry_t,
                div_lev_log_t,
                repo_t,
                surf_v[:, 0],
                surf_v[:, 1],
                surf_v[:, 2],
                surf_v[:, 3],
                surf_v[:, 4],
                surf_v[:, 5],
                surf_v[:, 6],
                dist_put,
                b_out.proxy_delta_weights,
                ones_d,
                role_a_t,
                role_b_t,
            ],
            dim=1,
        )

        log_fp_term = torch.log(f_p_term)
        cov_term = corr_t * v_term_v.unsqueeze(1) * v_term_v.unsqueeze(0) * off_diag
        # Smooth abs so jacfwd stays finite even on exact tied spots m0[0] == m0[1].
        diff_fp = log_fp_term.unsqueeze(1) - log_fp_term.unsqueeze(0)
        log_perf_diff = torch.sqrt(diff_fp * diff_fp + 1e-12) * off_diag
        edge_v = torch.stack(
            [corr_t * off_diag, cov_term, log_perf_diff, off_diag],
            dim=-1,
        )

        sched_v = torch.stack(
            [
                tau_obs_t,
                dtau_t,
                df_pay_t,
                log_ko_t,
                is_call_t,
                log_cpn_t,
                is_guar_t,
                cpn_r_t,
                torch.log(b_out.f_w.clamp(min=1e-12)),
                b_out.v_w,
                b_out.p_proxy,
                b_out.q_proxy,
            ],
            dim=1,
        )

        glob_v = glob_base.clone()
        glob_v[9] = b_out.l_put_proxy

        pv_p = _assemble_w1_pv_torch(
            b_out,
            df_pay_t,
            cpn_r_t,
            strike=k_put,
            option_gearing=g_put,
            c_mem_0=c_mem_0,
            is_memory=is_memory,
            no_notional_repayment=no_notional,
            autocall_coupon=ac_cpn,
            force_coupons_on_autocall=force_cpn,
            is_long_basket=is_long_basket,
        )
        return (
            node_v,
            edge_v,
            sched_v,
            glob_v,
            pv_p,
            b_out.p_proxy,
            b_out.q_proxy,
            b_out.l_put_proxy,
            b_out.f_w,
            b_out.v_w,
        )

    m0_t = torch.tensor(m0_np, dtype=torch.float64)
    with torch.no_grad():
        (
            node_act,
            edge_act,
            sched_act,
            glob_act,
            pv_proxy,
            p_proxy_v,
            q_proxy_v,
            l_put_proxy_v,
            f_w_v,
            v_w_v,
        ) = _features_from_m0(m0_t, use_fast_cpu=True)

        # Central finite-difference Jacobian wrt ln(M_0) using fast C++23 Tier-1 backend.
        h_fd = 1e-6
        inv_2h = 0.5 / h_fd
        ep = math.exp(h_fd)
        em = math.exp(-h_fd)
        jac_node_act = torch.zeros((d_active, d_active, 16), dtype=torch.float64)
        jac_edge_self_act = torch.zeros((d_active, d_active, 2), dtype=torch.float64)
        jac_sched_act = torch.zeros((d_active, m_rem, 4), dtype=torch.float64)
        jac_lput_act = torch.zeros((d_active,), dtype=torch.float64)
        delta_proxy_act = torch.zeros((d_active,), dtype=torch.float64)
        for i in range(d_active):
            m_up = m0_t.clone()
            m_dn = m0_t.clone()
            m_up[i] = m0_t[i] * ep
            m_dn[i] = m0_t[i] * em
            n_u, e_u, s_u, _, pv_u, _, _, lp_u, _, _ = _features_from_m0(m_up, use_fast_cpu=True)
            n_d, e_d, s_d, _, pv_d, _, _, lp_d, _, _ = _features_from_m0(m_dn, use_fast_cpu=True)
            jac_node_act[i] = (n_u - n_d) * inv_2h
            jac_edge_self_act[i] = (e_u[i, :, 1:3] - e_d[i, :, 1:3]) * inv_2h
            jac_sched_act[i] = (s_u[:, 8:12] - s_d[:, 8:12]) * inv_2h
            jac_lput_act[i] = (lp_u - lp_d) * inv_2h
            delta_proxy_act[i] = (pv_u - pv_d) * inv_2h

    # Precompute analytically bumped PROXY_KO_PROB at B_KO * exp(ko_bump_delta)
    ko_bumped = torch.where(
        is_call_t > 0.5,
        ko_t * math.exp(ko_bump_delta),
        ko_t,
    )
    ko_prob_b = _digital_call_spread_prob_torch(f_w_v, v_w_v, ko_bumped, ko_spread)
    h_b = is_call_t * ko_prob_b.clamp(0.0, 1.0 - 1e-7)
    surv_b = torch.cumprod(1.0 - h_b, dim=0)
    surv_prev_b = torch.cat([torch.ones(1, dtype=torch.float64), surv_b[:-1]], dim=0)
    p_proxy_ko_bumped_act = surv_prev_b * h_b

    # Pad to (d_max, m_max) for AutocallPreprocState and batch tensors
    node_pad = torch.zeros((1, d_max, 16), dtype=torch.float64)
    node_pad[0, :d_active] = node_act
    edge_pad = torch.zeros((1, d_max, d_max, 4), dtype=torch.float64)
    edge_pad[0, :d_active, :d_active] = edge_act
    amask_pad = torch.zeros((1, d_max), dtype=torch.float64)
    amask_pad[0, :d_active] = 1.0
    sched_pad = torch.zeros((1, m_max, 12), dtype=torch.float64)
    sched_pad[0, :m_rem] = sched_act
    smask_pad = torch.zeros((1, m_max), dtype=torch.float64)
    smask_pad[0, :m_rem] = 1.0
    glob_pad = glob_act.unsqueeze(0)

    jac_node_pad = torch.zeros((d_max, d_max, 16), dtype=torch.float64)
    jac_node_pad[:d_active, :d_active] = jac_node_act
    jac_edge_pad = torch.zeros((d_max, d_max, 2), dtype=torch.float64)
    jac_edge_pad[:d_active, :d_active] = jac_edge_self_act
    jac_sched_pad = torch.zeros((d_max, m_max, 4), dtype=torch.float64)
    jac_sched_pad[:d_active, :m_rem] = jac_sched_act
    jac_lput_pad = torch.zeros((d_max,), dtype=torch.float64)
    jac_lput_pad[:d_active] = jac_lput_act
    delta_proxy_pad = torch.zeros((d_max,), dtype=torch.float64)
    delta_proxy_pad[:d_active] = delta_proxy_act

    p_proxy_pad = torch.zeros((m_max,), dtype=torch.float64)
    p_proxy_pad[:m_rem] = p_proxy_v
    q_proxy_pad = torch.zeros((m_max,), dtype=torch.float64)
    q_proxy_pad[:m_rem] = q_proxy_v
    p_bumped_pad = torch.zeros((m_max,), dtype=torch.float64)
    p_bumped_pad[:m_rem] = p_proxy_ko_bumped_act

    # Construct lightweight AutocallPreprocState for GPU Dupire QMC
    obs_dates_dt = [_BASE_AS_OF + timedelta(days=int(round(t * 365.0))) for t in tau_obs_np]
    r_tot_obs_t = torch.full((d_active, m_rem), r_disc, dtype=torch.float64)
    df_fin_obs_t = df_pay_t.unsqueeze(0).expand(d_active, m_rem).clone()
    df_repo_obs_t = torch.exp(-torch.tensor(repo_avg_np[:, None] * tau_obs_np[None, :], dtype=torch.float64))
    fwd_price_t = torch.tensor(fwd_0_np, dtype=torch.float64)
    fwd_perf_t = torch.tensor(m0_np[:, None] * fwd_rel_np, dtype=torch.float64)
    iv_t = torch.tensor(iv_0_np, dtype=torch.float64)
    v_perf_t = iv_t * sqrt_tau_t.unsqueeze(0) * cash_lev_t

    div_out = EscrowedDividendOutput(
        identifiers=sorted_ids,
        as_of_date=_BASE_AS_OF,
        observation_dates=obs_dates_dt,
        div_pv=torch.zeros(d_active, dtype=torch.float64),
        escrowed_spot=torch.tensor(spots_np, dtype=torch.float64),
        forward_price=fwd_price_t,
        forward_rel=fwd_rel_t,
        forward_perf=fwd_perf_t,
        cash_div_leverage=cash_lev_t,
        escrowed_div_fraction=torch.zeros(d_active, dtype=torch.float64),
        df_fin_obs=df_fin_obs_t,
        df_repo_obs=df_repo_obs_t,
        r_tot_obs=r_tot_obs_t,
        repo_spread_avg=repo_t,
        dividend_events={tid: [] for tid in sorted_ids},
    )

    sentinel_res = SentinelBarrierResult(
        ko_barriers_raw=ko_barriers_np,
        cpn_barriers_raw=cpn_barriers_np,
        is_callable=is_callable_np,
        log_ko_barrier_feat=log_ko_feat_np,
        is_guaranteed_cpn=is_guar_np,
        log_cpn_barrier_feat=log_cpn_feat_np,
    )
    seasoned = SeasonedScheduleState(
        as_of_date=_BASE_AS_OF,
        start_date=_BASE_AS_OF - timedelta(days=30),
        payment_currency="RUB",
        day_count_convention="Act365",
        original_identifiers=sorted_ids,
        sorted_identifiers=sorted_ids,
        lex_perm=lex_perm_np,
        inv_lex_perm=inv_lex_perm_np,
        spots_t0=torch.tensor(spots_np, dtype=torch.float64),
        ref_spots=ref_t,
        moneyness_t0=m0_t,
        settled_indices=[],
        in_transit_indices=[],
        future_indices=list(range(m_rem)),
        c_mem_0=c_mem_0,
        i_ki_past=i_ki_past,
        already_autocalled=False,
        autocalled_period_index=None,
        pv_in_transit=0.0,
        future_obs_dates=obs_dates_dt,
        future_pay_dates=obs_dates_dt,
        tau_obs=tau_obs_np,
        delta_tau=delta_tau_np,
        tau_pay=tau_obs_np,
        future_coupon_rates=cpn_rates_np,
        sentinel_barriers=sentinel_res,
    )

    baseline_out = AutocallBaselineOutput(
        f_w=f_w_v,
        v_w=v_w_v,
        h_proxy=(p_proxy_v / torch.cat([torch.ones(1, dtype=torch.float64), (1.0 - torch.cumsum(p_proxy_v, 0))[:-1]]).clamp(min=1e-6)).clamp(0.0, 1.0),
        p_proxy=p_proxy_v,
        survival_proxy=(1.0 - torch.cumsum(p_proxy_v, dim=0)).clamp(min=0.0),
        q_proxy=q_proxy_v,
        l_put_proxy=l_put_proxy_v,
        proxy_delta_weights=node_act[:, int(AutocallNodeCol.PROXY_DELTA_WEIGHT)],
        option_pv_proxy=torch.tensor(0.0, dtype=torch.float64),
        notional_pv_proxy=torch.tensor(0.0, dtype=torch.float64),
        coupon_pv_proxy=torch.tensor(0.0, dtype=torch.float64),
        product_value_proxy=pv_proxy,
    )

    state = AutocallPreprocState(
        node_feats=node_pad,
        edge_feats=edge_pad,
        asset_mask=amask_pad,
        sched_feats=sched_pad,
        sched_mask=smask_pad,
        global_feats=glob_pad,
        d_assets=d_active,
        m_rem=m_rem,
        d_max=d_max,
        m_max=m_max,
        original_identifiers=sorted_ids,
        sorted_identifiers=sorted_ids,
        lex_perm=lex_perm_np,
        inv_lex_perm=inv_lex_perm_np,
        spots_t0=torch.tensor(spots_np, dtype=torch.float64),
        ref_spots=ref_t,
        moneyness_t0=m0_t,
        corr_matrix=corr_t,
        tau_obs=tau_obs_t,
        delta_tau=dtau_t,
        tau_pay=tau_obs_t,
        df_obs=df_pay_t,
        df_pay=df_pay_t,
        ko_barriers=ko_t,
        is_callable=is_call_t,
        cpn_barriers=cpn_b_t,
        is_guaranteed_cpn=is_guar_t,
        coupon_amounts=cpn_r_t,
        forward_rel=fwd_rel_t,
        forward_perf=fwd_perf_t,
        cash_div_leverage=cash_lev_t,
        iv_at_fwd=iv_t,
        v_perf=v_perf_t,
        seasoned_state=seasoned,
        dividend_schedule=div_out,
        vol_surfaces=reg_surfaces,
        surface_feature_jacobians={},
        iv_at_fwd_jacobians={},
        discount_curve=None,
        baseline=baseline_out,
        strike=k_put,
        option_gearing=g_put,
        lower_strike=lower_strike_val if lower_strike_val > 0.0 else None,
        ki_barrier=b_ki if has_ki else None,
        i_ki_past=i_ki_past,
        c_mem_0=c_mem_0,
        is_memory=is_memory,
        ko_spread=ko_spread,
        cpn_spread=cpn_spread,
        ki_spread=ki_spread,
        no_notional_repayment=no_notional,
        autocall_coupon=ac_cpn,
        force_coupons_on_autocall=force_cpn,
        allow_autocall_on_expiry=False,
        is_long_basket=is_long_basket,
        payment_currency="RUB",
        callability_index=call_idx,
        coupon_index=cpn_idx,
        option_index=opt_idx,
        funding_leg=None,
        pv_in_transit=0.0,
        already_autocalled=False,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )

    mod_vec = torch.tensor(
        [
            1.0 if no_notional else 0.0,
            float(ac_cpn),
            1.0 if force_cpn else 0.0,
            1.0 if is_long_basket else 0.0,
            float(lower_strike_val),
        ],
        dtype=torch.float64,
    )

    return (
        state,
        p_proxy_pad,
        q_proxy_pad,
        l_put_proxy_v,
        pv_proxy,
        delta_proxy_pad,
        p_bumped_pad,
        jac_node_pad,
        jac_edge_pad,
        jac_sched_pad,
        jac_lput_pad,
        mod_vec,
        surf_indices,
    )


def generate_synthetic_autocall_batch(
    batch_size: int,
    *,
    d_max: int = D_MAX_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
    m_rem_max: int = M_MAX_DEFAULT,
    n_mc_paths: int = 8192,
    n_substeps: int = 8,
    eta_lkj: float = 2.0,
    crisis_prob: float = 0.15,
    ko_bump_delta: float = 0.02,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    qmc_device: torch.device | str | None = None,
    fixed_d: Optional[int] = None,
    cv_mode: str = "none",
    compute_pathwise_delta: bool = True,
    domain_mode: str = "tier2_desk",
) -> SyntheticAutocallBatch:
    """Generates stratified Owen-Sobol v2.0 autocallable contracts labeled by GPU Dupire LV QMC."""
    _ = eta_lkj, crisis_prob
    if domain_mode not in ("tier2_desk", "full_stress"):
        raise ValueError(f"domain_mode must be 'tier2_desk' or 'full_stress', got {domain_mode!r}.")
    dev = torch.device(device)
    eff_qmc_dev = (
        torch.device(qmc_device)
        if qmc_device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    eff_m_rem_max = min(max(int(m_rem_max), 1), m_max)

    # Independent stratified Owen-Sobol draws per contract (eliminates modulo-index coupling).
    u_mat = _sample_owen_sobol_matrix(batch_size, 21, seed=seed)
    rng = np.random.default_rng(seed + 1337)

    # Stratify (d_active, m_rem) cleanly across buckets so GPU Dupire batching coalesces densely.
    if fixed_d is not None:
        d_choices = np.full(batch_size, max(1, min(int(fixed_d), d_max)), dtype=np.int64)
    else:
        d_choices = 1 + (np.arange(batch_size, dtype=np.int64) % d_max)
        rng.shuffle(d_choices)
        if domain_mode == "full_stress":
            pair_mask = (u_mat[:, 4] >= 0.82) & (u_mat[:, 4] < 0.94)
            d_choices = np.where(pair_mask & (d_max >= 2), 2, d_choices)

    m_grid = np.clip(
        1 + np.floor(u_mat[:, 19] * eff_m_rem_max).astype(np.int64),
        1,
        eff_m_rem_max,
    )
    # Quantize M_rem for large batches into rich bucket coverage [1..48] for high GPU occupancy.
    if batch_size >= 64 and eff_m_rem_max >= 12:
        m_palette = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48], dtype=np.int64)
        m_palette = m_palette[m_palette <= eff_m_rem_max]
        use_pal = u_mat[:, 20] < 0.85
        pal_idx = np.clip(np.floor(u_mat[:, 19] * len(m_palette)).astype(np.int64), 0, len(m_palette) - 1)
        m_choices = np.where(use_pal, m_palette[pal_idx], m_grid)
    else:
        m_choices = m_grid

    states: list[AutocallPreprocState] = []
    node_list: list[torch.Tensor] = []
    edge_list: list[torch.Tensor] = []
    amask_list: list[torch.Tensor] = []
    sched_list: list[torch.Tensor] = []
    smask_list: list[torch.Tensor] = []
    glob_list: list[torch.Tensor] = []

    p_px_list: list[torch.Tensor] = []
    q_px_list: list[torch.Tensor] = []
    lp_px_list: list[torch.Tensor] = []
    pv_px_list: list[torch.Tensor] = []
    dp_px_list: list[torch.Tensor] = []
    pb_px_list: list[torch.Tensor] = []
    jn_list: list[torch.Tensor] = []
    je_list: list[torch.Tensor] = []
    js_list: list[torch.Tensor] = []
    jl_list: list[torch.Tensor] = []
    mod_list: list[torch.Tensor] = []

    for b in range(batch_size):
        out_b = _build_contract_state_and_jacs(
            u_row=u_mat[b],
            d_active=int(d_choices[b]),
            m_rem=int(m_choices[b]),
            d_max=d_max,
            m_max=m_max,
            ko_bump_delta=ko_bump_delta,
            rng=rng,
            attach_surfaces=True,
            domain_mode=domain_mode,
        )
        st_b = out_b[0]
        states.append(st_b)
        node_list.append(st_b.node_feats[0])
        edge_list.append(st_b.edge_feats[0])
        amask_list.append(st_b.asset_mask[0])
        sched_list.append(st_b.sched_feats[0])
        smask_list.append(st_b.sched_mask[0])
        glob_list.append(st_b.global_feats[0])
        p_px_list.append(out_b[1])
        q_px_list.append(out_b[2])
        lp_px_list.append(out_b[3])
        pv_px_list.append(out_b[4])
        dp_px_list.append(out_b[5])
        pb_px_list.append(out_b[6])
        jn_list.append(out_b[7])
        je_list.append(out_b[8])
        js_list.append(out_b[9])
        jl_list.append(out_b[10])
        mod_list.append(out_b[11])

    solver = DupireQMCGpuSolver(device=eff_qmc_dev)
    qmc_cfg = DupireQMCConfig(
        n_paths=int(n_mc_paths),
        n_replicates=8 if int(n_mc_paths) >= 64 else 1,
        n_substeps=int(n_substeps),
        mode="dupire_lv",
        precision="mixed_fp32_fp64",
        cv_mode=cv_mode,  # type: ignore[arg-type]
        seed=int(seed) + 999,
    )
    pv_mc_t, p_mc_t, q_mc_t, l_put_mc_t, delta_mc_opt = solver.simulate_batch_for_training(
        states,
        config=qmc_cfg,
        compute_pathwise_delta=compute_pathwise_delta,
    )
    delta_mc_t = (
        delta_mc_opt
        if delta_mc_opt is not None
        else torch.zeros((batch_size, d_max), dtype=torch.float64)
    )

    mods = torch.stack(mod_list, dim=0)

    batch = SyntheticAutocallBatch(
        node_feats=torch.stack(node_list, dim=0),
        edge_feats=torch.stack(edge_list, dim=0),
        asset_mask=torch.stack(amask_list, dim=0),
        sched_feats=torch.stack(sched_list, dim=0),
        sched_mask=torch.stack(smask_list, dim=0),
        global_feats=torch.stack(glob_list, dim=0),
        p_proxy=torch.stack(p_px_list, dim=0),
        q_proxy=torch.stack(q_px_list, dim=0),
        l_put_proxy=torch.stack(lp_px_list, dim=0),
        pv_proxy=torch.stack(pv_px_list, dim=0),
        delta_proxy=torch.stack(dp_px_list, dim=0),
        p_mc=p_mc_t,
        q_mc=q_mc_t,
        l_put_mc=l_put_mc_t,
        pv_mc=pv_mc_t,
        delta_mc=delta_mc_t,
        no_notional_repayment=mods[:, 0],
        autocall_coupon=mods[:, 1],
        force_coupons_on_autocall=mods[:, 2],
        is_long_basket=mods[:, 3],
        lower_strike=mods[:, 4],
        p_proxy_ko_bumped=torch.stack(pb_px_list, dim=0),
        jac_node_logm=torch.stack(jn_list, dim=0),
        jac_edge_self=torch.stack(je_list, dim=0),
        jac_sched_logm=torch.stack(js_list, dim=0),
        jac_lput_logm=torch.stack(jl_list, dim=0),
    )
    return batch.to(device=dev, dtype=dtype)


def generate_dupire_autocall_dataset(
    n_contracts: int,
    *,
    d_max: int = D_MAX_DEFAULT,
    m_max: int = M_MAX_DEFAULT,
    n_mc_paths: int = 8192,
    n_substeps: int = 8,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    qmc_device: torch.device | str | None = None,
    cv_mode: str = "none",
) -> SyntheticAutocallBatch:
    """Convenience wrapper generating a v2.0 GPU Dupire QMC dataset of size `n_contracts`."""
    return generate_synthetic_autocall_batch(
        batch_size=n_contracts,
        d_max=d_max,
        m_max=m_max,
        m_rem_max=m_max,
        n_mc_paths=n_mc_paths,
        n_substeps=n_substeps,
        seed=seed,
        dtype=dtype,
        device=device,
        qmc_device=qmc_device,
        cv_mode=cv_mode,
        compute_pathwise_delta=True,
    )


__all__ = [
    "SyntheticAutocallBatch",
    "sample_lkj_cholesky_bartlett",
    "generate_synthetic_autocall_batch",
    "generate_dupire_autocall_dataset",
]
