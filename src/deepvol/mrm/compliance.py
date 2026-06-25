"""
compliance.py — SR 26-2 Compliance Layer
Implements Population Stability Index (PSI) online drift tracking,
OOD detection/clamping, and structured compliance logging.
"""

import numpy as np
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger("deepvol.compliance")

# FNO training ranges for Heston parameters
RH_BOUNDS = {
    "kappa": (0.5, 5.0),
    "theta": (0.01, 0.25),
    "sigma": (0.1, 1.5),
    "rho": (-0.95, 0.0),
    "v0": (0.01, 0.25),
    "H": (0.04, 0.15)
}

PARAM_NAMES = ["kappa", "theta", "sigma", "rho", "v0", "H"]


class DriftMonitor:
    """
    Tracks parameter drift over a rolling window using the Population Stability Index (PSI).
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: List[np.ndarray] = []

    def add_parameters(self, theta: np.ndarray):
        self.history.append(theta.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def compute_psi(self) -> Dict[str, float]:
        if len(self.history) < 20:
            # Need a minimum history size for stable binning
            return {}

        data = np.array(self.history)  # shape: (W, 6)
        psi_dict = {}

        for idx, name in enumerate(PARAM_NAMES):
            low, high = RH_BOUNDS[name]
            
            # Expected: Uniform reference distribution (10 bins -> 10% expected each)
            expected = np.full(10, 0.10)
            
            actual_data = data[:, idx]
            actual_freq, _ = np.histogram(actual_data, bins=10, range=(low, high))
            actual = actual_freq / len(actual_data)
            
            # Laplace smoothing to avoid log(0) or division by zero
            actual = (actual + 1e-5) / (1.0 + 10 * 1e-5)
            
            # Compute PSI formula: sum((Actual - Expected) * ln(Actual / Expected))
            psi = np.sum((actual - expected) * np.log(actual / expected))
            psi_dict[name] = float(psi)

            # Log warnings for significant or moderate drift
            if psi >= 0.25:
                logger.warning(
                    json.dumps({
                        "event": "PARAMETER_DRIFT_WARNING",
                        "parameter": name,
                        "psi": psi,
                        "status": "significant_drift"
                    })
                )
            elif psi >= 0.1:
                logger.info(
                    json.dumps({
                        "event": "PARAMETER_DRIFT_INFO",
                        "parameter": name,
                        "psi": psi,
                        "status": "moderate_drift"
                    })
                )

        return psi_dict


# Singleton drift monitor instance
_global_monitor = DriftMonitor()


def check_compliance(theta: np.ndarray) -> np.ndarray:
    """
    Inspects input parameters for OOD violations and logs any anomalies.
    Applies boundary clamping and feeds the parameters to the rolling drift monitor.
    """
    theta_clamped = theta.copy()
    ood_detected = False
    log_details = []

    for idx, name in enumerate(PARAM_NAMES):
        low, high = RH_BOUNDS[name]
        val = theta[idx]
        if val < low or val > high:
            ood_detected = True
            clamped_val = np.clip(val, low, high)
            theta_clamped[idx] = clamped_val
            log_details.append({
                "parameter": name,
                "value": float(val),
                "clamped_value": float(clamped_val),
                "bounds": [low, high]
            })

    if ood_detected:
        logger.warning(
            json.dumps({
                "event": "OOD_PARAMETER_DETECTION",
                "details": log_details
            })
        )

    _global_monitor.add_parameters(theta_clamped)
    _global_monitor.compute_psi()

    return theta_clamped
