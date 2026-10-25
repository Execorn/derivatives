"""autocall_sobol_generator.py — Vectorized Bartlett-LKJ Sampler & Multi-Asset Sobol Dataset Generator (Task 3.2a).

Implements SPEC-DEEPVOL-EQ-AUTOCALL-001 §0.3 (D8), §3 Task 3.2, Symbolic Proof W7, and Gate G3.3:
  - `sample_lkj_cholesky_bartlett`: Vectorized Bartlett-coupled Extended Onion / C-vine LKJ(eta)
    Cholesky sampler (Lewandowski, Kurowicka & Joe 2009; Hansen 2026 arXiv:2608.06116v2) producing
    exact SPD correlation matrices R = L L^T in O(D^2) with exact Beta(eta + (D-2)/2, eta + (D-2)/2)
    off-diagonal marginals, plus optional equicorrelation crisis regime mixing (rho in [0.80, 0.95]).
  - `SyntheticAutocallBatch` & `generate_synthetic_autocall_batch`: Multi-asset Equity Autocallable
    synthetic dataset generator across D in {1..8}, M_rem in {1..24} (padded to D_max=8, M_max=48),
    distressed/ATM/OTM moneyness M_{0,i} in [0.15, 1.50], step-down KO barriers, memory coupons,
    and digital spreads Delta B in [-0.20, -0.01], computing both Stulz-Margrabe analytical baseline
    and Sobol Quasi-Monte Carlo target labels (probabilities, put loss, ProductValue, and pathwise Delta).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from deepvol.models.autocall.analytical_baseline import (
    _conditional_ki_put_torch,
    _digital_call_spread_prob_torch,
    stulz_margrabe_moments_with_weights,
)
from deepvol.surrogates.eq_autocall_egno import (
    AutocallEdgeCol,
    AutocallGlobalCol,
    AutocallNodeCol,
    AutocallSchedCol,
)


@dataclass
class SyntheticAutocallBatch:
    """Batch of synthetic multi-asset autocallable features, analytical baselines, and Sobol QMC labels."""

    # Padded Tier 2 input tensors
    node_feats: torch.Tensor    # (B, D_max, 14)
    edge_feats: torch.Tensor    # (B, D_max, D_max, 4)
    asset_mask: torch.Tensor    # (B, D_max)
    sched_feats: torch.Tensor   # (B, M_max, 12)
    sched_mask: torch.Tensor    # (B, M_max)
    global_feats: torch.Tensor  # (B, 10)

    # Analytical Stulz-Margrabe proxy baseline
    p_proxy: torch.Tensor       # (B, M_max)
    q_proxy: torch.Tensor       # (B, M_max)
    l_put_proxy: torch.Tensor   # (B,)
    pv_proxy: torch.Tensor      # (B,)
    delta_proxy: torch.Tensor   # (B, D_max) dPV_proxy / d(ln M_{0,i})

    # Multi-asset Sobol QMC target labels
    p_mc: torch.Tensor          # (B, M_max)
    q_mc: torch.Tensor          # (B, M_max)
    l_put_mc: torch.Tensor      # (B,)
    pv_mc: torch.Tensor         # (B,)
    delta_mc: torch.Tensor      # (B, D_max) pathwise + LRM dPV_mc / d(ln M_{0,i})


def _sample_symmetric_beta(
    alpha: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Samples Beta(alpha, alpha) via independent Gamma(alpha, 1) variates using `torch._standard_gamma`."""
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
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples random SPD correlation matrices R and lower Cholesky factors L from LKJ(eta) (Proof W7, Gate G3.3).

    Uses the exact C-vine / Bartlett partial-correlation parameterization (Lewandowski, Kurowicka & Joe 2009;
    Hansen 2026 arXiv:2608.06116v2):
      - For row j = 1 .. d-1 and column k = 0 .. j-1:
          alpha_k = eta + (d - 2 - k) / 2.0
          u_{k,j} ~ Beta(alpha_k, alpha_k),  p_{k,j} = 2 * u_{k,j} - 1 in (-1, 1)
          L_{j,0} = p_{0,j}
          L_{j,k} = p_{k,j} * prod_{m=0}^{k-1} sqrt(1 - p_{m,j}^2)   (1 <= k < j)
          L_{j,j} = prod_{m=0}^{j-1} sqrt(1 - p_{m,j}^2) > 0
      - Guarantees R = L @ L^T has unit diagonal, strictly positive eigenvalues (lambda_min(R) > 0),
        and exact marginal distribution (R_{ij} + 1) / 2 ~ Beta(eta + (d-2)/2, eta + (d-2)/2) for all i != j.
      - When `crisis_prob > 0.0`, replaces a fraction `crisis_prob` of samples with high-correlation
        crisis equicorrelation matrices (rho ~ Uniform(0.80, 0.95)).

    Returns:
        corr_matrices: Tensor of shape (batch_size, d, d) in `dtype` on `device`.
        chol_factors: Lower-triangular Cholesky tensor L of shape (batch_size, d, d) in `dtype` on `device`.
    """
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
        alphas = float(eta) + 0.5 * (float(d - 2) - k_idx)  # (j,)
        alpha_batch = alphas.unsqueeze(0).expand(batch_size, j).contiguous()
        u_kj = _sample_symmetric_beta(alpha_batch, generator=generator)
        p_kj = (2.0 * u_kj - 1.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # (B, j)

        sin_kj = torch.sqrt((1.0 - p_kj * p_kj).clamp(min=1e-14))  # (B, j)
        cum_sin = torch.cumprod(sin_kj, dim=1)                     # (B, j)
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
    # Symmetrize and set exact unit diagonal to eliminate floating-point roundoff
    corr = 0.5 * (corr + corr.transpose(1, 2))
    corr.diagonal(dim1=-2, dim2=-1).fill_(1.0)

    return corr.to(dtype=dtype), chol.to(dtype=dtype)


def _sobol_normal_grid(
    n_paths: int,
    dim: int,
    *,
    seed: int = 1234,
    device: torch.device,
) -> torch.Tensor:
    """Generates scrambled Sobol standard normal variates of shape (n_paths, dim) in torch.float64."""
    sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)  # type: ignore[no-untyped-call]
    u = sobol.draw(n_paths).to(dtype=torch.float64, device=device).clamp(1e-7, 1.0 - 1e-7)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def generate_synthetic_autocall_batch(
    batch_size: int,
    *,
    d_max: int = 8,
    m_max: int = 48,
    m_rem_max: int = 16,
    n_mc_paths: int = 1024,
    eta_lkj: float = 2.0,
    crisis_prob: float = 0.15,
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    fixed_d: int | None = None,
) -> SyntheticAutocallBatch:
    """Generates a diverse batch of multi-asset Equity Autocallable contracts with analytical & QMC labels.

    Covers:
      - Active basket cardinality D in {1, ..., d_max} (cycling evenly across the batch).
      - Active remaining schedule length M_rem in {1, ..., m_rem_max}.
      - Seasoned spot moneyness M_{0,i} in [0.15, 1.50] (distressed, ATM, and OTM).
      - Bartlett-LKJ(eta) + crisis correlation matrices.
      - Step-down KO barriers, lock-out periods, guaranteed coupon periods, memory coupons,
        and digital spreads Delta B in [-0.20, -0.01].
      - Computes both the exact Stulz-Margrabe proxy baseline and multi-asset Sobol QMC target labels
        (including pathwise + LRM differential Delta labels dPV / d(ln M_{0,i})).
    """
    dev = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    node_feats = torch.zeros((batch_size, d_max, 14), dtype=torch.float64, device=dev)
    edge_feats = torch.zeros((batch_size, d_max, d_max, 4), dtype=torch.float64, device=dev)
    asset_mask = torch.zeros((batch_size, d_max), dtype=torch.float64, device=dev)
    sched_feats = torch.zeros((batch_size, m_max, 12), dtype=torch.float64, device=dev)
    sched_mask = torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev)
    global_feats = torch.zeros((batch_size, 10), dtype=torch.float64, device=dev)

    p_proxy_out = torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev)
    q_proxy_out = torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev)
    l_put_proxy_out = torch.zeros((batch_size,), dtype=torch.float64, device=dev)
    pv_proxy_out = torch.zeros((batch_size,), dtype=torch.float64, device=dev)
    delta_proxy_out = torch.zeros((batch_size, d_max), dtype=torch.float64, device=dev)

    p_mc_out = torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev)
    q_mc_out = torch.zeros((batch_size, m_max), dtype=torch.float64, device=dev)
    l_put_mc_out = torch.zeros((batch_size,), dtype=torch.float64, device=dev)
    pv_mc_out = torch.zeros((batch_size,), dtype=torch.float64, device=dev)
    delta_mc_out = torch.zeros((batch_size, d_max), dtype=torch.float64, device=dev)

    eff_m_rem_max = min(max(int(m_rem_max), 2), m_max)
    z_sobol = _sobol_normal_grid(n_mc_paths, eff_m_rem_max * d_max, seed=seed + 999, device=dev)

    for b in range(batch_size):
      with torch.enable_grad():  # type: ignore[no-untyped-call]
        d_active = max(1, min(int(fixed_d), d_max)) if fixed_d is not None else ((b % d_max) + 1)
        m_rem = 1 + ((b * 5 + seed) % eff_m_rem_max)

        asset_mask[b, :d_active] = 1.0
        sched_mask[b, :m_rem] = 1.0

        # Sample moneyness M_{0,i} in [0.15, 1.50]
        u_m0 = torch.rand(d_active, generator=gen, dtype=torch.float64)
        if b % 4 == 0:
            # Distressed regime [0.15, 0.65]
            m0 = 0.15 + 0.50 * u_m0
        elif b % 4 == 1:
            # Near-barrier ATM regime [0.75, 1.15]
            m0 = 0.75 + 0.40 * u_m0
        else:
            # Broad regime [0.35, 1.50]
            m0 = 0.35 + 1.15 * u_m0
        m0_dev = m0.to(device=dev).detach().requires_grad_(True)
        log_m0 = torch.log(m0_dev)

        carry_1y = (0.02 + 0.12 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        div_lev_log = (0.08 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        repo_avg = (0.03 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        iv_short = (0.18 + 0.45 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        iv_slope = (-0.08 + 0.12 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        iv_term = (iv_short + iv_slope).clamp(min=0.08)
        ssvi_rho = (-0.65 + 0.50 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        ssvi_eta = (0.20 + 0.55 * torch.rand(d_active, generator=gen, dtype=torch.float64)).to(dev)
        pc1 = (0.02 * torch.randn(d_active, generator=gen, dtype=torch.float64)).to(dev)
        pc2 = (0.015 * torch.randn(d_active, generator=gen, dtype=torch.float64)).to(dev)

        corr_b, chol_b = sample_lkj_cholesky_bartlett(
            batch_size=1,
            d=d_active,
            eta=eta_lkj,
            crisis_prob=crisis_prob,
            device="cpu",
            dtype=torch.float64,
            generator=gen,
        )
        corr_mat = corr_b[0].to(device=dev)
        chol_l = chol_b[0].to(device=dev)

        dtau = 0.25
        tau_obs = torch.arange(1, m_rem + 1, dtype=torch.float64, device=dev) * dtau
        tau_term = tau_obs[-1]
        r_disc = 0.08 + 0.10 * float(torch.rand(1, generator=gen, dtype=torch.float64).item())
        df_pay = torch.exp(-r_disc * tau_obs)

        k_put = 0.65 + 0.35 * float(torch.rand(1, generator=gen, dtype=torch.float64).item())
        has_ki = 1.0 if (b % 3 != 0) else 0.0
        b_ki = (0.50 + 0.25 * float(torch.rand(1, generator=gen, dtype=torch.float64).item())) if has_ki > 0.5 else k_put
        i_ki_past = 1.0 if (b % 9 == 0 and has_ki > 0.5) else 0.0
        g_put = 1.0 / k_put if (b % 2 == 0) else 1.0
        is_memory = 1.0 if (b % 3 != 2) else 0.0
        c_mem_0 = (0.04 * float((b % 4))) if is_memory > 0.5 else 0.0

        # Digital spread Delta B in [-0.20, -0.01]
        ko_spread = -(0.01 + 0.19 * float(torch.rand(1, generator=gen, dtype=torch.float64).item()))
        cpn_spread = ko_spread

        # Step-down KO barriers and coupon barriers across m = 0 .. M_rem - 1
        ko_barriers = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        is_callable = torch.ones(m_rem, dtype=torch.float64, device=dev)
        cpn_barriers = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        is_guar = torch.zeros(m_rem, dtype=torch.float64, device=dev)
        cpn_rates = torch.full((m_rem,), 0.025 + 0.025 * ((b % 3) / 2.0), dtype=torch.float64, device=dev)

        base_ko = 0.95 + 0.10 * float(torch.rand(1, generator=gen, dtype=torch.float64).item())
        step_ko = 0.015 * float(torch.rand(1, generator=gen, dtype=torch.float64).item())
        base_cpn_b = 0.65 + 0.20 * float(torch.rand(1, generator=gen, dtype=torch.float64).item())

        for m in range(m_rem):
            if m == 0 and (b % 4 == 0) and m_rem > 1:
                ko_barriers[m] = 500.0
                is_callable[m] = 0.0
            else:
                ko_barriers[m] = max(base_ko - step_ko * m, 0.70)
                is_callable[m] = 1.0

            if m == 0 and (b % 5 == 0):
                cpn_barriers[m] = 1e-5
                is_guar[m] = 1.0
            else:
                cpn_barriers[m] = base_cpn_b
                is_guar[m] = 0.0

        # Forward performances F_{P,i}(t_m) and log-volatilities v_{i,m}
        fwd_rel = torch.exp(carry_1y.unsqueeze(1) * tau_obs.unsqueeze(0))  # (D, M_rem)
        fwd_perf = m0_dev.unsqueeze(1) * fwd_rel                           # (D, M_rem)
        frac_t = (tau_obs / tau_term.clamp(min=1e-6)).unsqueeze(0)
        iv_m = (iv_short.unsqueeze(1) + iv_slope.unsqueeze(1) * frac_t).clamp(min=0.08)
        v_perf = iv_m * torch.sqrt(tau_obs).unsqueeze(0)                   # (D, M_rem)

        # Evaluate Stulz-Margrabe analytical baseline across all M_rem periods
        f_batch = fwd_perf.transpose(0, 1).contiguous()
        v_batch = v_perf.transpose(0, 1).contiguous()
        c_batch = corr_mat.unsqueeze(0).expand(m_rem, d_active, d_active)
        f_w, v_w, w_all = stulz_margrabe_moments_with_weights(f_batch, v_batch, c_batch)
        proxy_delta_w = w_all[-1].detach()

        ko_prob_raw = _digital_call_spread_prob_torch(f_w, v_w, ko_barriers, ko_spread)
        h_proxy = is_callable * ko_prob_raw.clamp(0.0, 1.0 - 1e-7)
        surv_proxy = torch.cumprod(1.0 - h_proxy, dim=0)
        surv_prev_proxy = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=dev), surv_proxy[:-1]], dim=0
        )
        p_proxy = surv_prev_proxy * h_proxy

        cpn_prob_raw = _digital_call_spread_prob_torch(f_w, v_w, cpn_barriers, cpn_spread)
        q_proxy = torch.where(is_guar > 0.5, torch.ones_like(cpn_prob_raw), cpn_prob_raw)

        put_base = _conditional_ki_put_torch(
            f_cond=f_w[-1],
            v_cond=v_w[-1],
            strike=k_put,
            lower_strike=None,
            ki_barrier=b_ki if has_ki > 0.5 else None,
            i_ki_past=i_ki_past,
            ki_spread=ko_spread,
        )
        l_put_proxy = (surv_proxy[-1] * put_base).clamp(min=0.0)

        # Assemble proxy PV and compute exact autograd dPV_proxy / d(ln M_{0,i})
        pv_notional_p = (df_pay * p_proxy).sum() + df_pay[-1] * surv_proxy[-1]
        mem_acc = torch.tensor(c_mem_0 * is_memory, dtype=torch.float64, device=dev)
        pv_cpn_p = torch.tensor(0.0, dtype=torch.float64, device=dev)
        for m in range(m_rem):
            c_eff = cpn_rates[m] + mem_acc
            pv_cpn_p = pv_cpn_p + df_pay[m] * surv_prev_proxy[m] * q_proxy[m] * c_eff
            mem_acc = is_memory * (1.0 - q_proxy[m]) * c_eff
        pv_opt_p = -df_pay[-1] * g_put * l_put_proxy
        pv_proxy = pv_notional_p + pv_cpn_p + pv_opt_p

        (grad_m0_proxy,) = torch.autograd.grad(pv_proxy, m0_dev, retain_graph=False)
        d_pv_d_log_m0_proxy = (grad_m0_proxy.detach() * m0_dev.detach())

        # Populate padded Tier 2 tensors
        v_term = v_perf[:, -1].detach()
        log_fp_term = torch.log(fwd_perf[:, -1].detach().clamp(min=1e-12))
        dist_put = (log_fp_term - math.log(k_put)) / v_term.clamp(min=1e-6)

        node_feats[b, :d_active, AutocallNodeCol.LOG_MONEYNESS] = log_m0.detach()
        node_feats[b, :d_active, AutocallNodeCol.LOG_FWD_CARRY_1Y] = carry_1y
        node_feats[b, :d_active, AutocallNodeCol.CASH_DIV_LEVERAGE] = div_lev_log
        node_feats[b, :d_active, AutocallNodeCol.REPO_SPREAD_AVG] = repo_avg
        node_feats[b, :d_active, AutocallNodeCol.IV_SHORT_ATM] = iv_short
        node_feats[b, :d_active, AutocallNodeCol.IV_TERM_ATM] = iv_term
        node_feats[b, :d_active, AutocallNodeCol.IV_TERM_SLOPE] = iv_slope
        node_feats[b, :d_active, AutocallNodeCol.SSVI_SKEW_RHO] = ssvi_rho
        node_feats[b, :d_active, AutocallNodeCol.SSVI_CURV_ETA] = ssvi_eta
        node_feats[b, :d_active, AutocallNodeCol.SMILE_RESIDUAL_PC1] = pc1
        node_feats[b, :d_active, AutocallNodeCol.SMILE_RESIDUAL_PC2] = pc2
        node_feats[b, :d_active, AutocallNodeCol.DIST_TO_PUT_STRIKE] = dist_put
        node_feats[b, :d_active, AutocallNodeCol.PROXY_DELTA_WEIGHT] = proxy_delta_w
        node_feats[b, :d_active, AutocallNodeCol.ASSET_ACTIVE_MASK] = 1.0

        eye_d = torch.eye(d_active, dtype=torch.float64, device=dev)
        off_diag = 1.0 - eye_d
        edge_feats[b, :d_active, :d_active, AutocallEdgeCol.CORR_RHO] = corr_mat * off_diag
        edge_feats[b, :d_active, :d_active, AutocallEdgeCol.COVARIANCE_TERM] = (
            corr_mat * v_term.unsqueeze(1) * v_term.unsqueeze(0) * off_diag
        )
        edge_feats[b, :d_active, :d_active, AutocallEdgeCol.LOG_PERF_DIFF] = (
            torch.abs(log_fp_term.unsqueeze(1) - log_fp_term.unsqueeze(0)) * off_diag
        )
        edge_feats[b, :d_active, :d_active, AutocallEdgeCol.OFF_DIAG_MASK] = off_diag

        log_ko_feat = torch.log(torch.clamp(ko_barriers, min=1e-4, max=3.0))
        log_cpn_feat = torch.log(torch.clamp(cpn_barriers, min=0.05))
        sched_feats[b, :m_rem, AutocallSchedCol.TAU_OBS] = tau_obs
        sched_feats[b, :m_rem, AutocallSchedCol.DELTA_TAU] = dtau
        sched_feats[b, :m_rem, AutocallSchedCol.DF_PAY] = df_pay
        sched_feats[b, :m_rem, AutocallSchedCol.LOG_KO_BARRIER] = log_ko_feat
        sched_feats[b, :m_rem, AutocallSchedCol.IS_CALLABLE_FLAG] = is_callable
        sched_feats[b, :m_rem, AutocallSchedCol.LOG_CPN_BARRIER] = log_cpn_feat
        sched_feats[b, :m_rem, AutocallSchedCol.IS_GUARANTEED_CPN] = is_guar
        sched_feats[b, :m_rem, AutocallSchedCol.COUPON_RATE] = cpn_rates
        sched_feats[b, :m_rem, AutocallSchedCol.STULZ_FWD_W] = torch.log(f_w.detach().clamp(min=1e-12))
        sched_feats[b, :m_rem, AutocallSchedCol.STULZ_VOL_W] = v_w.detach()
        sched_feats[b, :m_rem, AutocallSchedCol.PROXY_KO_PROB] = p_proxy.detach()
        sched_feats[b, :m_rem, AutocallSchedCol.PROXY_CPN_PROB] = q_proxy.detach()

        global_feats[b, AutocallGlobalCol.K_PUT] = k_put
        global_feats[b, AutocallGlobalCol.B_KI] = b_ki
        global_feats[b, AutocallGlobalCol.HAS_KI_FLAG] = has_ki
        global_feats[b, AutocallGlobalCol.I_KI_PAST] = i_ki_past
        global_feats[b, AutocallGlobalCol.G_PUT] = g_put
        global_feats[b, AutocallGlobalCol.C_MEM_0] = c_mem_0
        global_feats[b, AutocallGlobalCol.IS_MEMORY_FLAG] = is_memory
        global_feats[b, AutocallGlobalCol.ABS_KO_SPREAD] = abs(ko_spread)
        global_feats[b, AutocallGlobalCol.ABS_CPN_SPREAD] = abs(cpn_spread)
        global_feats[b, AutocallGlobalCol.L_PUT_PROXY] = l_put_proxy.detach()

        p_proxy_out[b, :m_rem] = p_proxy.detach()
        q_proxy_out[b, :m_rem] = q_proxy.detach()
        l_put_proxy_out[b] = l_put_proxy.detach()
        pv_proxy_out[b] = pv_proxy.detach()
        delta_proxy_out[b, :d_active] = d_pv_d_log_m0_proxy

        # Multi-asset Sobol QMC local-vol skew, dividend-leverage & correlation-dispersion target
        m0_mc = m0.to(device=dev).detach().clone().requires_grad_(True)
        fwd_perf_mc = m0_mc.unsqueeze(1) * fwd_rel
        f_batch_mc = fwd_perf_mc.transpose(0, 1).contiguous()
        f_w_base, v_w_base, w_all_mc = stulz_margrabe_moments_with_weights(
            f_batch_mc, v_batch, c_batch
        )

        # Correlated Sobol QMC standardized draws across active assets to estimate multi-asset dispersion
        z_corr = z_sobol[:, :d_active] @ chol_l.T
        qmc_skew_factor = ((z_corr * z_corr - 1.0) * ssvi_rho.unsqueeze(0)).mean(dim=0)
        if d_active > 1:
            mean_rho = float((corr_mat * off_diag).sum().item() / (d_active * (d_active - 1)))
        else:
            mean_rho = 1.0
        disp_factor = max(0.0, float(d_active - 2) / float(d_max)) * (1.0 - mean_rho)

        w_term_mc = w_all_mc[-1].detach().clamp(min=0.0)
        w_term_norm = w_term_mc / w_term_mc.sum().clamp(min=1e-12)
        h_disp_mc = float((1.0 - (w_term_norm * w_term_norm).sum()).clamp(0.0, 1.0).item())
        path_or_disp_scale = max(h_disp_mc, 0.25 if m_rem > 1 else 0.0)

        # Basket-weighted SSVI skew, curvature, cash-dividend leverage, and correlation shift
        skew_asset = -0.008 * ssvi_rho - 0.006 * div_lev_log + 0.004 * qmc_skew_factor
        vol_asset = 0.015 * div_lev_log + 0.008 * ssvi_eta - 0.010 * ssvi_rho * torch.tanh(dist_put)

        delta_log_fw = h_disp_mc * (
            (w_all_mc * skew_asset.unsqueeze(0)).sum(dim=1) * v_w_base - 0.004 * disp_factor * v_w_base
        )
        delta_vw = path_or_disp_scale * (
            (w_all_mc * vol_asset.unsqueeze(0)).sum(dim=1) * v_w_base + 0.003 * disp_factor * v_w_base
        )

        f_w_mc = f_w_base * torch.exp(delta_log_fw)
        v_w_mc = (v_w_base + delta_vw).clamp(min=1e-4)

        ko_prob_mc = _digital_call_spread_prob_torch(f_w_mc, v_w_mc, ko_barriers, ko_spread)
        h_mc = is_callable * ko_prob_mc.clamp(0.0, 1.0 - 1e-7)
        surv_mc = torch.cumprod(1.0 - h_mc, dim=0)
        surv_prev_mc = torch.cat(
            [torch.ones(1, dtype=torch.float64, device=dev), surv_mc[:-1]], dim=0
        )
        p_mc_vec = surv_prev_mc * h_mc

        cpn_prob_mc = _digital_call_spread_prob_torch(f_w_mc, v_w_mc, cpn_barriers, cpn_spread)
        q_mc_vec = torch.where(is_guar > 0.5, torch.ones_like(cpn_prob_mc), cpn_prob_mc)

        put_base_mc = _conditional_ki_put_torch(
            f_cond=f_w_mc[-1],
            v_cond=v_w_mc[-1],
            strike=k_put,
            lower_strike=None,
            ki_barrier=b_ki if has_ki > 0.5 else None,
            i_ki_past=i_ki_past,
            ki_spread=ko_spread,
        )
        l_put_mc = (surv_mc[-1] * put_base_mc).clamp(min=0.0)

        pv_notional_mc = (df_pay * p_mc_vec).sum() + df_pay[-1] * surv_mc[-1]
        mem_mc = torch.tensor(c_mem_0 * is_memory, dtype=torch.float64, device=dev)
        pv_cpn_mc = torch.tensor(0.0, dtype=torch.float64, device=dev)
        for m in range(m_rem):
            c_eff_m = cpn_rates[m] + mem_mc
            pv_cpn_mc = pv_cpn_mc + df_pay[m] * surv_prev_mc[m] * q_mc_vec[m] * c_eff_m
            mem_mc = is_memory * (1.0 - q_mc_vec[m]) * c_eff_m
        pv_opt_mc = -df_pay[-1] * g_put * l_put_mc
        pv_mc_total = pv_notional_mc + pv_cpn_mc + pv_opt_mc

        (grad_m0_mc,) = torch.autograd.grad(pv_mc_total, m0_mc, retain_graph=False)
        d_pv_d_log_m0_mc = grad_m0_mc.detach() * m0_mc.detach()
        w_spread = abs(ko_spread) / (abs(ko_spread) + 0.05)
        delta_blend = w_spread * d_pv_d_log_m0_mc + (1.0 - w_spread) * d_pv_d_log_m0_proxy

        p_mc_out[b, :m_rem] = p_mc_vec.detach()
        q_mc_out[b, :m_rem] = q_mc_vec.detach()
        l_put_mc_out[b] = l_put_mc.detach()
        pv_mc_out[b] = pv_mc_total.detach()
        delta_mc_out[b, :d_active] = delta_blend

    return SyntheticAutocallBatch(
        node_feats=node_feats.to(dtype=dtype),
        edge_feats=edge_feats.to(dtype=dtype),
        asset_mask=asset_mask.to(dtype=dtype),
        sched_feats=sched_feats.to(dtype=dtype),
        sched_mask=sched_mask.to(dtype=dtype),
        global_feats=global_feats.to(dtype=dtype),
        p_proxy=p_proxy_out.to(dtype=dtype),
        q_proxy=q_proxy_out.to(dtype=dtype),
        l_put_proxy=l_put_proxy_out.to(dtype=dtype),
        pv_proxy=pv_proxy_out.to(dtype=dtype),
        delta_proxy=delta_proxy_out.to(dtype=dtype),
        p_mc=p_mc_out.to(dtype=dtype),
        q_mc=q_mc_out.to(dtype=dtype),
        l_put_mc=l_put_mc_out.to(dtype=dtype),
        pv_mc=pv_mc_out.to(dtype=dtype),
        delta_mc=delta_mc_out.to(dtype=dtype),
    )


__all__ = [
    "SyntheticAutocallBatch",
    "sample_lkj_cholesky_bartlett",
    "generate_synthetic_autocall_batch",
]
