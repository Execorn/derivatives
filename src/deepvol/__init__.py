"""
DeepVol Option Pricing, Calibration, and Hedging Framework.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

__version__ = "1.0.0"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "calibrate": ("deepvol.calibration.interface", "calibrate"),
    "CalibrationResult": ("deepvol.calibration.interface", "CalibrationResult"),
    "compute_greeks": ("deepvol.greeks.interface", "compute_greeks"),
    "MLSVEngine": ("deepvol.models.mlsv_gpu", "MLSVEngine"),
    "RatesSABREngine": ("deepvol.models.sabr_rates", "RatesSABREngine"),
    "SchwartzSmithEngine": ("deepvol.models.schwartz_smith", "SchwartzSmithEngine"),
    "HestonEngine": ("deepvol.models.heston", "HestonEngine"),
    "rBergomiEngine": ("deepvol.models.rbergomi_gpu", "rBergomiEngine"),
}

__all__ = [
    "__version__",
    "calibrate",
    "CalibrationResult",
    "compute_greeks",
    "MLSVEngine",
    "RatesSABREngine",
    "SchwartzSmithEngine",
    "HestonEngine",
    "rBergomiEngine",
]


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
