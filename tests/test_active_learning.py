import torch
import os
import numpy as np
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import IVSurfaceNormalizer
from deepvol.calibration.active_learning import (
    ParameterNormalizerGrey,
    make_spatial_input,
)


def test_load_trained_grey_model():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(
        project_root, "artifacts", "weights", "fno_grey_final_prod.pth"
    )
    param_norm_path = os.path.join(
        project_root, "artifacts", "models", "param_normalizer_grey.npz"
    )
    iv_norm_path = os.path.join(
        project_root, "artifacts", "models", "iv_normalizer_grey.npz"
    )

    # Verify files exist
    assert os.path.exists(weights_path), "fno_grey_final_prod.pth not found"
    assert os.path.exists(param_norm_path), "param_normalizer_grey.npz not found"
    assert os.path.exists(iv_norm_path), "iv_normalizer_grey.npz not found"

    # Load normalizers
    param_norm = ParameterNormalizerGrey.load(param_norm_path)
    iv_norm = IVSurfaceNormalizer.load(iv_norm_path)

    # Initialize model
    model = MirrorPaddedFNO2d(modes1=8, modes2=5, param_dim=5)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Verify inference
    params = torch.tensor([[0.04, 0.1, 1.5, -0.7, 0.85]], dtype=torch.float32)
    params_norm = torch.tensor(
        param_norm.transform(params.numpy()), dtype=torch.float32
    )

    T_grid = torch.tensor(
        [0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=torch.float32
    )
    K_grid = torch.tensor(
        [
            np.log(0.8),
            np.log(0.85),
            np.log(0.9),
            np.log(0.95),
            0.0,
            np.log(1.05),
            np.log(1.1),
            np.log(1.15),
            np.log(1.2),
        ],
        dtype=torch.float32,
    )

    spatial = make_spatial_input(T_grid, K_grid, "cpu")

    with torch.no_grad():
        pred_norm = model(spatial, params_norm)
        pred_iv = iv_norm.inverse_transform_tensor(pred_norm)

    assert pred_iv.shape == (1, 8, 9)
    assert (pred_iv > 0.0).all()
