from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "NeuralSDE": ("deepvol.models.neural_sde", "NeuralSDE"),
    "NeuralSDEPricer": ("deepvol.models.neural_sde", "NeuralSDEPricer"),
    "compute_calibration_loss": (
        "deepvol.models.neural_sde",
        "compute_calibration_loss",
    ),
    "schwartz_smith_price_black76": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_price_black76",
    ),
    "schwartz_smith_price_fourier": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_price_fourier",
    ),
    "schwartz_smith_price_black76_pt": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_price_black76_pt",
    ),
    "schwartz_smith_price_fourier_pt": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_price_fourier_pt",
    ),
    "schwartz_smith_price_cos": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_price_cos",
    ),
    "schwartz_smith_price_cos_pt": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_price_cos_pt",
    ),
    "schwartz_smith_cf": ("deepvol.models.schwartz_smith", "schwartz_smith_cf"),
    "schwartz_smith_cf_pt": (
        "deepvol.models.schwartz_smith",
        "schwartz_smith_cf_pt",
    ),
    "calibrate_schwartz_smith": (
        "deepvol.models.schwartz_smith",
        "calibrate_schwartz_smith",
    ),
    "run_kalman_filter": (
        "deepvol.models.schwartz_smith",
        "run_kalman_filter",
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
