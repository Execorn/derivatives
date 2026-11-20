"""
autocall — Multi-Asset & Vanilla Equity Autocallable Pricing Package.

Preserves backward-compatible exports from legacy `deepvol.models.autocall`
(`make_obs_indices`, `autocall_upper_bound`, `_price_autocall_kernel`, `price_autocall_mc`)
while lazily exposing multi-asset autocall submodules via PEP 562.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def make_obs_indices(n_obs: int, T: float, N_steps: int) -> List[int]:
    """
    Returns a sorted list of step indices (ints) where observation dates fall,
    uniformly spaced over (0, T].
    """
    dt = T / N_steps
    indices: List[int] = []
    for i in range(1, n_obs + 1):
        t_obs_i = i * T / n_obs
        idx_i = int(round(t_obs_i / dt))
        idx_i = max(1, min(N_steps, idx_i))
        indices.append(idx_i)
    return sorted(indices)


def autocall_upper_bound(n_obs: int, coupon: float, T: float, r: float) -> float:
    """
    Analytical upper bound price assuming spot always above barrier (called at first observation date):
        upper_bound = exp(-r * t_1) * (1 + coupon * t_1)
    where t_1 = T / n_obs.
    """
    t_1 = T / n_obs
    return math.exp(-r * t_1) * (1.0 + coupon * t_1)


@torch.compile(mode="reduce-overhead")
def _price_autocall_kernel(
    S: torch.Tensor,
    obs_indices: torch.Tensor,
    B: torch.Tensor,
    coupon: torch.Tensor,
    r: torch.Tensor,
    T: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compiled GPU kernel for autocall Monte Carlo payoff accumulation.
    Returns cloned outputs to prevent CUDAGraphs static buffer corruption.
    """
    B_batch = S.shape[0]
    N_paths = S.shape[1]
    S0 = S[:, :, 0]

    called = torch.zeros((B_batch, N_paths), dtype=torch.bool, device=S.device)
    npv = torch.zeros((B_batch, N_paths), dtype=torch.float64, device=S.device)
    life = torch.zeros((B_batch, N_paths), dtype=torch.float64, device=S.device)

    B_mat = B.view(-1, 1)
    coupon_mat = coupon.view(-1, 1)
    r_mat = r.view(-1, 1)

    for idx in obs_indices:
        t_i = idx.to(torch.float64) * dt
        trigger = (S[:, :, idx] >= (B_mat * S0)) & (~called)
        discount = torch.exp(-r_mat * t_i)
        npv = npv + trigger.to(torch.float64) * discount * (1.0 + coupon_mat * t_i)
        life = life + trigger.to(torch.float64) * t_i
        called = called | trigger

    # Capital protection for uncalled paths at maturity
    uncalled = ~called
    discount_T = torch.exp(-r_mat * T)
    npv = npv + uncalled.to(torch.float64) * discount_T
    life = life + uncalled.to(torch.float64) * T

    mean_npv = npv.mean(dim=1)
    call_prob = called.to(torch.float64).mean(dim=1)
    exp_life = life.mean(dim=1)

    return mean_npv.clone(), call_prob.clone(), exp_life.clone()


def price_autocall_mc(
    S: torch.Tensor,
    obs_indices: Union[List[int], torch.Tensor],
    B: torch.Tensor,
    coupon: torch.Tensor,
    r: torch.Tensor,
    T: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Monte Carlo pricing for 1-leg vanilla autocallable notes.
    """
    if isinstance(obs_indices, list):
        obs_tensor = torch.tensor(obs_indices, dtype=torch.long, device=S.device)
    else:
        obs_tensor = obs_indices.to(device=S.device, dtype=torch.long)

    S_f64 = S.to(torch.float64)
    B_f64 = B.to(device=S.device, dtype=torch.float64).view(-1)
    coupon_f64 = coupon.to(device=S.device, dtype=torch.float64).view(-1)
    r_f64 = r.to(device=S.device, dtype=torch.float64).view(-1)

    return _price_autocall_kernel(
        S_f64, obs_tensor, B_f64, coupon_f64, r_f64, float(T), float(dt)
    )


_LAZY_SUBMODULE_EXPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    "FEATURE_SCHEMA_VERSION": ("deepvol.models.autocall.feature_layout", "FEATURE_SCHEMA_VERSION"),
    "D_MAX_DEFAULT": ("deepvol.models.autocall.feature_layout", "D_MAX_DEFAULT"),
    "M_MAX_DEFAULT": ("deepvol.models.autocall.feature_layout", "M_MAX_DEFAULT"),
    "MonitoringIndexCode": ("deepvol.models.autocall.feature_layout", "MonitoringIndexCode"),
    "AutocallNodeCol": ("deepvol.models.autocall.feature_layout", "AutocallNodeCol"),
    "AutocallEdgeCol": ("deepvol.models.autocall.feature_layout", "AutocallEdgeCol"),
    "AutocallSchedCol": ("deepvol.models.autocall.feature_layout", "AutocallSchedCol"),
    "AutocallGlobalCol": ("deepvol.models.autocall.feature_layout", "AutocallGlobalCol"),
    "AutocallBaselineOutput": ("deepvol.models.autocall.analytical_baseline", "AutocallBaselineOutput"),
    "compute_analytical_autocall_baseline": (
        "deepvol.models.autocall.analytical_baseline",
        "compute_analytical_autocall_baseline",
    ),
    "fast_stulz_margrabe_cpu": ("deepvol.models.autocall.analytical_baseline", "fast_stulz_margrabe_cpu"),
    "stulz_margrabe_moments_with_weights": (
        "deepvol.models.autocall.analytical_baseline",
        "stulz_margrabe_moments_with_weights",
    ),
    "stulz_margrabe_worst_of_moments": (
        "deepvol.models.autocall.analytical_baseline",
        "stulz_margrabe_worst_of_moments",
    ),
    "DupireQMCResult": ("deepvol.models.autocall.dupire_qmc_solver", "DupireQMCResult"),
    "RegularizedVolSurface": ("deepvol.models.autocall.dupire_qmc_solver", "RegularizedVolSurface"),
    "enforce_no_arbitrage_projection": (
        "deepvol.models.autocall.dupire_qmc_solver",
        "enforce_no_arbitrage_projection",
    ),
    "price_eq_autocall_dupire_qmc": (
        "deepvol.models.autocall.dupire_qmc_solver",
        "price_eq_autocall_dupire_qmc",
    ),
    "price_eq_autocall_qmc_fallback": (
        "deepvol.models.autocall.dupire_qmc_solver",
        "price_eq_autocall_qmc_fallback",
    ),
    "EscrowedDividendOutput": ("deepvol.models.autocall.escrowed_dividend", "EscrowedDividendOutput"),
    "ParsedDividendEvent": ("deepvol.models.autocall.escrowed_dividend", "ParsedDividendEvent"),
    "build_discount_rate_evaluator": (
        "deepvol.models.autocall.escrowed_dividend",
        "build_discount_rate_evaluator",
    ),
    "build_repo_rate_evaluator": ("deepvol.models.autocall.escrowed_dividend", "build_repo_rate_evaluator"),
    "compute_escrowed_dividend_forwards": (
        "deepvol.models.autocall.escrowed_dividend",
        "compute_escrowed_dividend_forwards",
    ),
    "merge_dividend_schedules": ("deepvol.models.autocall.escrowed_dividend", "merge_dividend_schedules"),
    "AutocallPostprocOutput": ("deepvol.models.autocall.pre_post_processor", "AutocallPostprocOutput"),
    "AutocallPreprocState": ("deepvol.models.autocall.pre_post_processor", "AutocallPreprocState"),
    "assemble_autocall_cashflows_and_pv": (
        "deepvol.models.autocall.pre_post_processor",
        "assemble_autocall_cashflows_and_pv",
    ),
    "preprocess_eq_autocall": ("deepvol.models.autocall.pre_post_processor", "preprocess_eq_autocall"),
    "CashFlowItem": ("deepvol.models.autocall.seasoned_collapser", "CashFlowItem"),
    "HistoricalPeriodRecord": ("deepvol.models.autocall.seasoned_collapser", "HistoricalPeriodRecord"),
    "SeasonedScheduleState": ("deepvol.models.autocall.seasoned_collapser", "SeasonedScheduleState"),
    "SentinelBarrierResult": ("deepvol.models.autocall.seasoned_collapser", "SentinelBarrierResult"),
    "canonicalize_asset_order": ("deepvol.models.autocall.seasoned_collapser", "canonicalize_asset_order"),
    "collapse_seasoned_schedule": (
        "deepvol.models.autocall.seasoned_collapser",
        "collapse_seasoned_schedule",
    ),
    "evaluate_monitoring_index_np": (
        "deepvol.models.autocall.seasoned_collapser",
        "evaluate_monitoring_index_np",
    ),
    "evaluate_monitoring_index_torch": (
        "deepvol.models.autocall.seasoned_collapser",
        "evaluate_monitoring_index_torch",
    ),
    "normalize_sentinel_barriers": (
        "deepvol.models.autocall.seasoned_collapser",
        "normalize_sentinel_barriers",
    ),
    "greeks_engine": ("deepvol.models.autocall.greeks_engine", None),
    "AutocallGreeksResult": ("deepvol.models.autocall.greeks_engine", "AutocallGreeksResult"),
    "compute_eq_autocall_greeks": ("deepvol.models.autocall.greeks_engine", "compute_eq_autocall_greeks"),
    "compute_batched_surrogate_greeks": (
        "deepvol.models.autocall.greeks_engine",
        "compute_batched_surrogate_greeks",
    ),
}

__all__ = [
    "make_obs_indices",
    "autocall_upper_bound",
    "_price_autocall_kernel",
    "price_autocall_mc",
    *_LAZY_SUBMODULE_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    target = _LAZY_SUBMODULE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
