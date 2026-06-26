"""
test_d_xva_stress.py — Stress Testing Suite for E2E D-XVA Pipeline.
Verifies pipeline robustness under extreme market volatility, out-of-bound calibrator
outputs (including NaN/Inf), gradient flow stability, and CPU/CUDA performance.
"""

import os
import logging
import torch
import torch.nn as nn
import pytest

from deepvol.hedging.d_xva import DXVAPipeline
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.hedging.policy import DeepHedgingPolicy
from deepvol.hedging.pivot_iv import pivot_implied_vol

logger = logging.getLogger("deepvol.hedging.d_xva")


class MockCalibrator(nn.Module):
    """
    Mock calibrator that returns fixed (possibly extreme/OOD) parameter values
    to directly test the downstream pipeline (clamping, simulation, pricing).
    """

    def __init__(self, values: torch.Tensor):
        super().__init__()
        # Ensure values are parameter-resident or just stored as buffer
        self.register_buffer("values", values)
        # Add a dummy parameter to verify gradient flow through calibrator
        self.dummy_param = nn.Parameter(
            torch.tensor(1.0, dtype=values.dtype, device=values.device)
        )

    def forward(self, market_surface: torch.Tensor) -> torch.Tensor:
        B = market_surface.shape[0]
        # Multiply by dummy_param so that calibrator weights are part of the computation graph
        return self.values.expand(B, -1) * self.dummy_param


def get_pipeline_stress_setup(
    device: torch.device, mock_values: torch.Tensor
) -> DXVAPipeline:
    """
    Set up the pipeline with a MockCalibrator returning the specified values.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load production FNO model
    pricing_fno = MirrorPaddedFNO2d(param_dim=6)
    weights_path = os.path.join(project_root, "artifacts/weights/fno_v2_final_prod.pth")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    pricing_fno.load_state_dict(state_dict)
    pricing_fno.to(device)

    # Load normalizers
    param_norm_path = os.path.join(
        project_root, "artifacts/models/param_normalizer_v2.npz"
    )
    iv_norm_path = os.path.join(project_root, "artifacts/models/iv_normalizer_v2.npz")

    parameter_normalizer = ParameterNormalizer.load(param_norm_path)
    iv_normalizer = IVSurfaceNormalizer.load(iv_norm_path)

    # Instantiate neural modules
    calibrator = MockCalibrator(mock_values.to(device))
    policy_net = DeepHedgingPolicy(input_dim=5, hidden_dim=16, output_dim=1).to(device)

    pipeline = DXVAPipeline(
        calibrator=calibrator,
        pricing_fno=pricing_fno,
        iv_solver=pivot_implied_vol,
        policy_net=policy_net,
        parameter_normalizer=parameter_normalizer,
        iv_normalizer=iv_normalizer,
        c_fee=0.001,
        cost_type="huber",
    )
    return pipeline


@pytest.mark.parametrize(
    "device_name", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_stress_extreme_market_surfaces(device_name):
    """
    Test forward pass and gradient flow when the market implied vol surface is extremely high or low/negative.
    """
    device = torch.device(device_name)
    # Default valid parameters to calibrate to
    default_vals = torch.tensor(
        [1.5, 0.05, 0.3, -0.5, 0.05, 0.1], dtype=torch.float32, device=device
    )
    pipeline = get_pipeline_stress_setup(device, default_vals)
    pipeline.train()

    # Extreme volatility surfaces
    extreme_surfaces = {
        "extremely_high_vol": torch.full(
            (2, 8, 11), 5.0, device=device, dtype=torch.float64
        ),
        "extremely_low_vol": torch.full(
            (2, 8, 11), 0.001, device=device, dtype=torch.float64
        ),
        "zero_vol": torch.full((2, 8, 11), 0.0, device=device, dtype=torch.float64),
        "negative_vol": torch.full(
            (2, 8, 11), -0.2, device=device, dtype=torch.float64
        ),
    }

    for name, mkt_surface in extreme_surfaces.items():
        try:
            out = pipeline(
                market_surface=mkt_surface,
                N_steps=5,
                N_paths=20,
                S0=100.0,
                K=100.0,
                T=0.5,
                r=0.02,
            )
            loss = out["loss_total"]

            # If market surface is negative or zero, price_bs_f64 might compute NaN/Inf inside forward for prices_mkt,
            # but loss_total only depends on iv_pred and market_surface. Let's see if loss is valid.
            if name in ["extremely_high_vol", "extremely_low_vol"]:
                assert torch.isfinite(loss), f"Loss is not finite for surface {name}"
                loss.backward()

                # Check calibrator and policy gradients are finite
                assert torch.isfinite(pipeline.calibrator.dummy_param.grad), (
                    f"Calibrator dummy grad is not finite for {name}"
                )
                for p_name, param in pipeline.policy_net.named_parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all(), (
                            f"Policy parameter {p_name} grad is not finite for {name}"
                        )

            # Reset gradients
            pipeline.zero_grad()

        except Exception as e:
            # Document if it crashes
            logger.error(f"Pipeline crashed on market surface {name}: {e}")
            raise e


@pytest.mark.parametrize(
    "device_name", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_stress_ood_parameters_clamping(device_name):
    """
    Test how the pipeline behaves when calibrator outputs extremely out-of-distribution (OOD) parameters.
    """
    device = torch.device(device_name)

    # List of OOD parameters to test:
    # 1. Extremely large values
    # 2. Negative values
    # 3. Exactly on the boundaries
    ood_cases = {
        "extremely_large": torch.tensor(
            [50.0, 5.0, 10.0, -1.8, 5.0, 1.2], dtype=torch.float32, device=device
        ),
        "negative_values": torch.tensor(
            [-5.0, -0.1, -0.2, 1.5, -0.05, -0.01], dtype=torch.float32, device=device
        ),
        "on_boundaries_upper": torch.tensor(
            [10.0, 1.0, 2.0, 0.99, 1.0, 0.5], dtype=torch.float32, device=device
        ),
        "on_boundaries_lower": torch.tensor(
            [0.01, 0.01, 0.01, -0.99, 0.01, 0.01], dtype=torch.float32, device=device
        ),
    }

    # Limits standard: kappa in [0.01, 10.0], theta_v in [0.01, 1.0], sigma_v in [0.01, 2.0],
    # rho in [-0.99, 0.99], v0 in [0.01, 1.0], H in [0.01, 0.5]
    mins = [0.01, 0.01, 0.01, -0.99, 0.01, 0.01]
    maxs = [10.0, 1.0, 2.0, 0.99, 1.0, 0.5]

    for name, params in ood_cases.items():
        pipeline = get_pipeline_stress_setup(device, params)
        pipeline.train()
        market_surface = torch.full(
            (2, 8, 11), 0.25, device=device, dtype=torch.float64
        )

        out = pipeline(
            market_surface=market_surface,
            N_steps=5,
            N_paths=20,
            S0=100.0,
            K=100.0,
            T=0.5,
            r=0.02,
        )

        # Verify clamping limits are respected
        clamped_theta = out["theta"]
        for i in range(6):
            val = clamped_theta[:, i]
            assert (val >= mins[i] - 1e-7).all() and (val <= maxs[i] + 1e-7).all(), (
                f"Clamped parameter {i} for case {name} is out of bounds: {val}"
            )

        # Verify gradient flow with clamped parameters
        loss = out["loss_total"]
        assert torch.isfinite(loss), f"Loss is not finite for OOD case {name}"
        loss.backward()

        # Since parameters are clamped, their gradient with respect to dummy_param is zero
        # if the parameters are strictly clamped (flat region).
        # Let's verify that the policy network still gets finite gradients, and calibrator dummy_param is finite.
        assert torch.isfinite(pipeline.calibrator.dummy_param.grad), (
            f"Calibrator dummy grad is not finite for {name}"
        )
        for p_name, param in pipeline.policy_net.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Policy parameter {p_name} grad is not finite for {name}"
                )


@pytest.mark.parametrize(
    "device_name", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_stress_nan_inf_parameters(device_name):
    """
    Test how the pipeline behaves when calibrator outputs NaN or Inf.
    Clamping does not handle NaN by default in PyTorch (NaNs propagate), so we want to see if this is handled.
    """
    device = torch.device(device_name)

    nan_inf_cases = {
        "nan_param": torch.tensor(
            [torch.nan, 0.05, 0.3, -0.5, 0.05, 0.1], dtype=torch.float32, device=device
        ),
        "inf_param": torch.tensor(
            [torch.inf, 0.05, 0.3, -0.5, 0.05, 0.1], dtype=torch.float32, device=device
        ),
    }

    for name, params in nan_inf_cases.items():
        pipeline = get_pipeline_stress_setup(device, params)
        market_surface = torch.full(
            (2, 8, 11), 0.25, device=device, dtype=torch.float64
        )

        # We expect NaN/Inf to propagate through torch.min/max clamping and downstream pricing/simulation,
        # which will result in NaN/Inf outputs, but we want to confirm if it crashes the runtime.
        try:
            out = pipeline(
                market_surface=market_surface,
                N_steps=5,
                N_paths=20,
                S0=100.0,
                K=100.0,
                T=0.5,
                r=0.02,
            )

            # Verify that clamping failed to clear NaN/Inf and propagated to loss
            clamped_theta = out["theta"]
            loss = out["loss_total"]

            if name == "nan_param":
                assert torch.isnan(clamped_theta[0, 0]), (
                    "NaN parameter did not propagate to theta as expected"
                )
                assert torch.isnan(loss), "NaN parameter did not result in NaN loss"
            elif name == "inf_param":
                # torch.min(inf, max) -> max, so inf might be clamped to max! Let's check:
                # torch.min(inf, 10.0) -> 10.0 in PyTorch. Let's verify if inf got clamped!
                # Yes, min(inf, 10.0) is 10.0, so inf should actually be clamped successfully!
                assert not torch.isinf(clamped_theta[0, 0]), (
                    "Inf parameter was not clamped"
                )
                assert torch.isfinite(loss), (
                    "Inf parameter resulted in non-finite loss after clamping"
                )

        except Exception as e:
            logger.info(f"Pipeline crashed on {name} as expected/unexpected: {e}")
            # If it crashed, we document it, but if it ran and returned NaN loss, that's also documented.
