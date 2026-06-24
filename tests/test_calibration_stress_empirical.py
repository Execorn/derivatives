import os
import sys
import time
import pytest
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from deepvol.app.components.models import (
    reconstruct_sabr_surface,
    reconstruct_heston_surface,
    load_fno_model
)
from deepvol.calibration.interface import calibrate, CalibrationResult

class TestCalibrationStressEmpirical:
    def test_sabr_surface_extreme_parameters(self):
        """Verify SABR surface raises ValueError for invalid/out-of-bound parameters."""
        with pytest.raises(ValueError):
            reconstruct_sabr_surface(alpha=-0.1, rho=-0.4, nu=0.4)
        with pytest.raises(ValueError):
            reconstruct_sabr_surface(alpha=0.2, rho=-1.5, nu=0.4)
        with pytest.raises(ValueError):
            reconstruct_sabr_surface(alpha=0.2, rho=0.4, nu=-0.1)

    def test_heston_surface_extreme_parameters(self):
        """Verify Heston surface handles invalid parameters by throwing ValueError from pricing engine."""
        # Pricing under Heston with negative parameters raises ValueError in the characteristic function
        with pytest.raises(ValueError):
            reconstruct_heston_surface(kappa=-1.0, theta=0.05, sigma=0.3, rho=-0.6, v0=0.05)
        with pytest.raises(ValueError):
            reconstruct_heston_surface(kappa=2.0, theta=0.05, sigma=0.0, rho=-0.6, v0=0.05)
        with pytest.raises(ValueError):
            reconstruct_heston_surface(kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6, v0=0.0)

    def test_calibration_degenerate_surfaces(self):
        """Verify FNO-based Newton calibration behavior on degenerate/corrupt target surfaces.
        """
        model = load_fno_model("Classic Heston")
        if model is None:
            from deepvol.calibration.interface import _get_default_model
            model = _get_default_model("heston", torch.device("cpu"))
            
        T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
        K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)

        target_nan = np.full((8, 11), np.nan, dtype=np.float32)
        res = calibrate(
            market_iv_surface=target_nan,
            model_name="heston",
            method="newton",
            T_grid=T_grid,
            K_grid=K_grid,
            model=model,
            max_iter=2
        )
        assert res.status == "failed"
        assert np.isnan(res.rmse) or res.rmse == 0.0

    def test_calibration_throughput(self):
        """High-throughput scenario: Calibrate multiple surfaces sequentially and check latency.
        
        Moves FNO model to CUDA device if available to test GPU acceleration throughput.
        """
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        
        model = load_fno_model("Classic Heston")
        if model is None:
            from deepvol.calibration.interface import _get_default_model
            model = _get_default_model("heston", device)
        else:
            model.to(device)
            
        T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
        K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        
        # Pre-generate 10 different target surfaces
        rng = np.random.default_rng(42)
        surfaces = []
        for _ in range(10):
            kappa = rng.uniform(1.0, 3.0)
            theta = rng.uniform(0.02, 0.10)
            sigma = rng.uniform(0.1, 0.5)
            rho = rng.uniform(-0.8, -0.2)
            v0 = rng.uniform(0.02, 0.10)
            surf = reconstruct_heston_surface(kappa, theta, sigma, rho, v0)
            surfaces.append(surf)
            
        t0 = time.time()
        for surf in surfaces:
            # We use low starts/iterations to keep throughput high-throughput test quick
            res = calibrate(
                market_iv_surface=surf,
                model_name="heston",
                method="newton",
                T_grid=T_grid,
                K_grid=K_grid,
                model=model,
                device=device_name,
                max_iter=3,
                n_starts=1
            )
            # Ensure it is a valid result
            assert isinstance(res, CalibrationResult)
            
        elapsed = time.time() - t0
        avg_time_ms = (elapsed / 10) * 1000.0
        print(f"Average Newton-FNO Heston calibration time on {device_name.upper()}: {avg_time_ms:.2f}ms")
        
        # GPU calibration should be extremely fast (< 150ms)
        # If running on CPU, we expect < 350ms per run since starts/iterations are reduced
        latency_bound = 150.0 if device_name == "cuda" else 350.0
        assert avg_time_ms < latency_bound, f"Average calibration time on {device_name.upper()} too slow: {avg_time_ms:.2f}ms"
