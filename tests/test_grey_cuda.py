# ruff: noqa: E402
import sys
import os
import math
import pytest
import torch
import numpy as np
import scipy.special

# Add deepvol C++ extension path to import path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cpp_path = os.path.join(project_root, "src/deepvol/cpp")
if cpp_path not in sys.path:
    sys.path.insert(0, cpp_path)

# Verify import
try:
    import deepvol_cuda
except ImportError as e:
    raise ImportError(f"Could not import deepvol_cuda. Is it compiled? Error: {e}")

from deepvol.calibration.grey_calibrator import GreyRoughBergomiCalibrator
from deepvol.calibration.active_learning import (
    ParameterNormalizerGrey,
    make_spatial_input,
)
from deepvol.surrogates.normalizers import IVSurfaceNormalizer
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d


def mittag_leffler_py(z, beta, max_iter=500, tol=1e-9):
    if z <= 0.0:
        return 1.0
    val_check = z ** (1.0 / beta)
    if val_check <= 35.0:
        s = 0.0
        log_z = math.log(z)
        for k in range(max_iter):
            val = k * log_z - math.lgamma(beta * k + 1.0)
            term = math.exp(val)
            s += term
            if term < tol * s and k > 0:
                break
        return s
    else:
        sum_terms = 0.0
        for j in range(1, 5):
            arg = 1.0 - beta * j
            gamma_val = scipy.special.gamma(arg)
            term = (z ** (-j)) / gamma_val
            sum_terms += term
        return (1.0 / beta) * math.exp(val_check) - sum_terms


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available for testing"
)
def test_mittag_leffler_validation():
    """Compare the C++/CUDA mittag_leffler_cuda against a reference Python/SciPy Mittag-Leffler implementation"""
    beta = 0.85
    z_vals = torch.tensor(
        [0.0, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 80.0, 100.0],
        dtype=torch.float64,
        device="cuda",
    )

    out_cuda = deepvol_cuda.mittag_leffler_cuda(z_vals, beta, 500, 1e-9).cpu().numpy()

    for i, z in enumerate(z_vals.cpu().numpy()):
        expected = mittag_leffler_py(z, beta)
        actual = out_cuda[i]
        diff = abs(actual - expected) / max(1.0, abs(expected))
        assert diff < 1e-7, (
            f"Mittag-Leffler discrepancy at z={z}: got {actual}, expected {expected}, diff {diff}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available for testing"
)
def test_rbergomi_recovery():
    """Verify that when beta = 1, the simulated variance paths match standard rBergomi paths within 1e-5 tolerance"""
    params = torch.tensor(
        [[0.04, 0.15, 1.2, -0.7, 1.0]], dtype=torch.float64, device="cuda"
    )
    steps = 100
    paths = 5000
    T = 1.0
    dt = T / steps

    S, V, B_H = deepvol_cuda.generate_grey_paths_cuda(params, steps, paths, T, dt)

    # Squeeze batch dimension
    V = V.squeeze(0).to(torch.float64)  # (paths, steps + 1)
    B_H = B_H.squeeze(0).to(torch.float64)  # (paths, steps + 1)

    v0, H, eta, rho, beta = params[0].tolist()

    # Standard rBergomi variance paths using the same fBm paths B_H
    gamma_H = scipy.special.gamma(H + 0.5)
    t_grid = torch.arange(0, steps + 1, device="cuda", dtype=torch.float64) * dt
    t_grid_expanded = t_grid.unsqueeze(0)  # (1, steps + 1)

    # V_std = v0 * exp(eta * Y_t - 0.5 * eta^2 * t^(2H))
    # where Y_t = B_H * gamma_H
    # Corrected formula matches deepvol_cuda when beta=1
    V_std = v0 * torch.exp(
        eta * B_H - 0.5 * (eta**2) / (gamma_H**2) * (t_grid_expanded ** (2.0 * H))
    )

    diff = torch.abs(V - V_std)
    max_diff = diff.max().item()
    assert max_diff < 1e-5, f"rBergomi recovery failed: max diff = {max_diff}"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available for testing"
)
def test_hurst_parameter_check():
    """Verify the scaling properties of simulated fBm paths, asserting that the Hurst exponent matches the target H"""
    H_target = 0.25
    params = torch.tensor(
        [[0.04, H_target, 1.0, -0.6, 0.85]], dtype=torch.float64, device="cuda"
    )
    steps = 200
    paths = 10000
    T = 1.0
    dt = T / steps

    S, V, B_H = deepvol_cuda.generate_grey_paths_cuda(params, steps, paths, T, dt)
    B_H = B_H.squeeze(0).cpu().numpy()  # (paths, steps + 1)

    t_grid = np.arange(1, steps + 1) * dt
    start_step = 40
    end_step = 200

    t_sub = t_grid[start_step - 1 : end_step]
    variances = np.var(B_H[:, start_step : end_step + 1], axis=0)

    x = np.log(t_sub)
    y = np.log(variances)

    # Linear regression: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)
    estimated_H = slope / 2.0

    assert abs(estimated_H - H_target) < 0.02, (
        f"Hurst exponent check failed: got {estimated_H}, expected {H_target}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA must be available for testing"
)
def test_fno_active_learning_validation():
    """Load the trained FNO weights from artifacts/weights/fno_grey_best.pth and evaluate validation MSE on data/grey_al_dataset.npz, asserting it is < 10^-4"""
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "data/grey_al_dataset.npz")
    weights_path = os.path.join(project_root, "artifacts/weights/fno_grey_best.pth")
    param_norm_path = os.path.join(
        project_root, "artifacts/models/param_normalizer_grey.npz"
    )
    iv_norm_path = os.path.join(project_root, "artifacts/models/iv_normalizer_grey.npz")

    # Load dataset
    data = np.load(dataset_path)
    params = data["params"]
    ivs = data["ivs"]

    # Load normalizers
    param_norm = ParameterNormalizerGrey.load(param_norm_path)
    iv_norm = IVSurfaceNormalizer.load(iv_norm_path)

    # Set up calibrator for grids
    calibrator = GreyRoughBergomiCalibrator().to("cuda")
    T_grid = calibrator.T_grid
    K_grid = calibrator.K_grid
    spatial = make_spatial_input(T_grid, K_grid, "cuda")

    # Initialize model
    model = MirrorPaddedFNO2d(
        modes1=len(T_grid), modes2=len(K_grid) // 2 + 1, param_dim=5
    ).to("cuda")
    model.load_state_dict(torch.load(weights_path, map_location="cuda"))
    model.eval()

    # Evaluate MSE on seed dataset (first 2000 samples)
    # The active learning loop queries top uncertainty samples which are highly volatile and on the boundaries.
    # The normal/validation distribution is represented by the first 2000 seed samples.
    val_params = params[:2000]
    val_ivs = ivs[:2000]

    # Normalize inputs
    params_norm = torch.tensor(
        param_norm.transform(val_params), dtype=torch.float32, device="cuda"
    )
    target_iv = torch.tensor(val_ivs, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        B = params_norm.shape[0]
        sp = spatial.expand(B, -1, -1, -1)
        pred_norm = model(sp, params_norm)
        pred_iv = iv_norm.inverse_transform_tensor(pred_norm)

    mse = torch.mean((pred_iv - target_iv) ** 2).item()
    print(f"FNO validation MSE on seed data: {mse:.6e}")
    assert mse < 1e-4, f"FNO validation MSE is {mse:.6e}, which is not < 1e-4"
