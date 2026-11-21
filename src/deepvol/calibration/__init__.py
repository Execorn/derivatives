"""
src/calibration package — Joint calibration and batch calibration for
Rough Heston FNO surrogate.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

_LAZY_EXPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    "joint_calibration": ("deepvol.calibration.joint_calibration", None),
    "batch_calibration": ("deepvol.calibration.batch_calibration", None),
    "GreyRoughBergomiCalibrator": (
        "deepvol.calibration.grey_calibrator",
        "GreyRoughBergomiCalibrator",
    ),
    "run_active_learning": (
        "deepvol.calibration.active_learning",
        "run_active_learning",
    ),
}

__all__ = [
    "joint_calibration",
    "batch_calibration",
    "GreyRoughBergomiCalibrator",
    "run_active_learning",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    module = importlib.import_module(mod_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(__all__))
