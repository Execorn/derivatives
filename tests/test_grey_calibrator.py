import pytest
import torch
from deepvol.calibration.grey_calibrator import GreyRoughBergomiCalibrator


def test_grey_calibrator_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    calib = GreyRoughBergomiCalibrator(steps=50, paths=1000).to("cuda")

    # 2D params (v0, H, eta, rho, beta)
    params = torch.tensor(
        [[0.04, 0.1, 1.5, -0.7, 0.85], [0.09, 0.15, 2.0, -0.5, 0.9]],
        device="cuda",
        dtype=torch.float64,
    )

    # Price options
    prices = calib.price_surface(params, steps=50, N_paths=1000)
    assert prices.shape == (2, 8, 9)
    assert prices.dtype == torch.float64
    assert prices.device.type == "cuda"

    # Forward to IV
    ivs = calib(params)
    assert ivs.shape == (2, 8, 9)
    assert ivs.dtype == torch.float32
    assert ivs.device.type == "cuda"
    assert (ivs >= 0.01).all()
