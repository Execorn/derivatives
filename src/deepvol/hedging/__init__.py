from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "HedgingPolicy": ("deepvol.hedging.deep_hedging", "HedgingPolicy"),
    "DeepHedgingEnv": ("deepvol.hedging.deep_hedging", "DeepHedgingEnv"),
    "train_deep_hedger": ("deepvol.hedging.deep_hedging", "train_deep_hedger"),
    "estimate_gpd_tail_index_pwm": (
        "deepvol.hedging.deep_hedging",
        "estimate_gpd_tail_index_pwm",
    ),
    "compute_acf_loss": ("deepvol.hedging.deep_hedging", "compute_acf_loss"),
    "compute_leverage_loss": (
        "deepvol.hedging.deep_hedging",
        "compute_leverage_loss",
    ),
    "compute_cfvc_loss": ("deepvol.hedging.deep_hedging", "compute_cfvc_loss"),
    "BarrierHedgingEnv": (
        "deepvol.hedging.barrier_hedging",
        "BarrierHedgingEnv",
    ),
    "WGAN_GP_Generator": (
        "deepvol.hedging.adversarial_market",
        "WGAN_GP_Generator",
    ),
    "WGAN_GP_Discriminator": (
        "deepvol.hedging.adversarial_market",
        "WGAN_GP_Discriminator",
    ),
    "StylizedFactsAlignmentGAN": (
        "deepvol.hedging.adversarial_market",
        "StylizedFactsAlignmentGAN",
    ),
    "train_robust_minimax_hedger": (
        "deepvol.hedging.adversarial_market",
        "train_robust_minimax_hedger",
    ),
    "convert_returns_to_prices": (
        "deepvol.hedging.adversarial_market",
        "convert_returns_to_prices",
    ),
    "FrictionalHedgingEnv": (
        "deepvol.hedging.frictional_env",
        "FrictionalHedgingEnv",
    ),
    "IndifferencePricingEngine": (
        "deepvol.hedging.indifference_pricing",
        "IndifferencePricingEngine",
    ),
    "invert_implied_volatility_hybrid": (
        "deepvol.hedging.indifference_pricing",
        "invert_implied_volatility_hybrid",
    ),
    "compute_bs_greeks": ("deepvol.hedging.backtest", "compute_bs_greeks"),
    "get_whalley_wilmott_beta": (
        "deepvol.hedging.backtest",
        "get_whalley_wilmott_beta",
    ),
    "attribute_pnl_daily": ("deepvol.hedging.backtest", "attribute_pnl_daily"),
    "extract_empirical_option_path": (
        "deepvol.hedging.backtest",
        "extract_empirical_option_path",
    ),
    "run_empirical_backtest": (
        "deepvol.hedging.backtest",
        "run_empirical_backtest",
    ),
    "MartingaleViolationError": (
        "deepvol.hedging.mp_diffusion",
        "MartingaleViolationError",
    ),
    "PathDenoisingNet": ("deepvol.hedging.mp_diffusion", "PathDenoisingNet"),
    "project_spot_martingale": (
        "deepvol.hedging.mp_diffusion",
        "project_spot_martingale",
    ),
    "audit_martingale_paths": (
        "deepvol.hedging.mp_diffusion",
        "audit_martingale_paths",
    ),
    "MPDDPM": ("deepvol.hedging.mp_diffusion", "MPDDPM"),
    "PIVOTImpliedVolFunction": (
        "deepvol.hedging.pivot_iv",
        "PIVOTImpliedVolFunction",
    ),
    "pivot_implied_vol": ("deepvol.hedging.pivot_iv", "pivot_implied_vol"),
    "DeepHedgingPolicy": ("deepvol.hedging.policy", "DeepHedgingPolicy"),
    "proportional_transaction_cost": (
        "deepvol.hedging.policy",
        "proportional_transaction_cost",
    ),
    "huber_transaction_cost": (
        "deepvol.hedging.policy",
        "huber_transaction_cost",
    ),
    "sqrt_transaction_cost": (
        "deepvol.hedging.policy",
        "sqrt_transaction_cost",
    ),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
