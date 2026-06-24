"""
DeepVol Option Pricing, Calibration, and Hedging Framework.
"""
from __future__ import annotations

__version__ = "1.0.0"

# Main Developer APIs
from deepvol.calibration.interface import calibrate, CalibrationResult
from deepvol.greeks.interface import compute_greeks

# Core Engine classes for direct developer access
from deepvol.models.mlsv_gpu import MLSVEngine
from deepvol.models.sabr_rates import RatesSABREngine
from deepvol.models.schwartz_smith import SchwartzSmithEngine
from deepvol.models.heston import HestonEngine
from deepvol.models.rbergomi_gpu import rBergomiEngine

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
