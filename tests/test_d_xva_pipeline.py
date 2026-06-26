"""
test_d_xva_pipeline.py — Localized Verification Suite for E2E D-XVA Pipeline.
Verifies CPU/CUDA execution, differentiable gradient flow, low-vega gradient gating,
and compliance logging.
"""

import os
import logging
import torch
import torch.nn as nn

from deepvol.hedging.d_xva import DXVAPipeline
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
from deepvol.surrogates.normalizers import ParameterNormalizer, IVSurfaceNormalizer
from deepvol.hedging.policy import DeepHedgingPolicy
from deepvol.hedging.pivot_iv import pivot_implied_vol

# Setup logging capturing for verification
logger = logging.getLogger("deepvol.hedging.d_xva")


class SimpleNeuralCalibrator(nn.Module):
    """
    Differentiable MLP mapper from market IV surface (8x11) to Heston parameters (6).
    """

    def __init__(self, param_dim: int = 6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8 * 11, 32), nn.ReLU(), nn.Linear(32, param_dim)
        )

    def forward(self, market_surface: torch.Tensor) -> torch.Tensor:
        B = market_surface.shape[0]
        flat_surface = market_surface.view(B, -1).float()
        out = self.fc(flat_surface)

        # Scale output to be in reasonable parameter ranges
        # [kappa, theta_v, sigma_v, rho, v0, H]
        kappa = 1.0 + 0.5 * torch.tanh(out[:, 0:1])  # ~1.0
        theta_v = 0.05 + 0.02 * torch.tanh(out[:, 1:2])  # ~0.05
        sigma_v = 0.3 + 0.1 * torch.tanh(out[:, 2:3])  # ~0.3
        rho = -0.5 + 0.2 * torch.tanh(out[:, 3:4])  # ~-0.5
        v0 = 0.05 + 0.02 * torch.tanh(out[:, 4:5])  # ~0.05

        if out.shape[1] == 6:
            H = 0.1 + 0.05 * torch.tanh(out[:, 5:6])  # ~0.1
            return torch.cat([kappa, theta_v, sigma_v, rho, v0, H], dim=-1)
        else:
            return torch.cat([kappa, theta_v, sigma_v, rho, v0], dim=-1)


def get_pipeline_setup(device: torch.device):
    """
    Helper to set up pipeline components with production weights.
    """
    # Load production FNO model
    pricing_fno = MirrorPaddedFNO2d(param_dim=6)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(project_root, "artifacts/weights/fno_v2_final_prod.pth")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    pricing_fno.load_state_dict(state_dict)
    pricing_fno.to(device)

    # Load normalizers (Rough Heston v2 normalizers corresponding to fno_v2)
    param_norm_path = os.path.join(
        project_root, "artifacts/models/param_normalizer_v2.npz"
    )
    iv_norm_path = os.path.join(project_root, "artifacts/models/iv_normalizer_v2.npz")

    parameter_normalizer = ParameterNormalizer.load(param_norm_path)
    iv_normalizer = IVSurfaceNormalizer.load(iv_norm_path)

    # Instantiate neural modules
    calibrator = SimpleNeuralCalibrator(param_dim=6).to(device)
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


def test_d_xva_pipeline_cpu_and_cuda():
    """
    Test E2E forward pass and loss computation on both CPU and CUDA (if available).
    """
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        pipeline = get_pipeline_setup(device)

        # Prepare mock inputs
        market_surface = torch.full(
            (2, 8, 11), 0.25, device=device, dtype=torch.float64
        )  # flat 25% IV

        out = pipeline(
            market_surface=market_surface,
            N_steps=10,
            N_paths=50,
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.03,
            q=0.01,
        )

        assert "loss_total" in out
        assert "loss_cal" in out
        assert "loss_hedge" in out
        assert "loss_trans" in out
        assert not torch.isnan(out["loss_total"])
        assert out["loss_total"] > 0


def test_d_xva_gradient_flow():
    """
    Verify that gradients successfully flow back to both calibrator and policy network parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = get_pipeline_setup(device)
    pipeline.train()

    market_surface = torch.full((2, 8, 11), 0.25, device=device, dtype=torch.float64)

    out = pipeline(
        market_surface=market_surface,
        N_steps=5,
        N_paths=20,
        S0=100.0,
        K=100.0,
        T=0.5,
        r=0.02,
    )

    loss = out["loss_total"]
    loss.backward()

    # Check calibrator gradients
    calib_has_grad = False
    for param in pipeline.calibrator.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), "Calibrator grad contains NaN"
            assert not torch.isinf(param.grad).any(), "Calibrator grad contains Inf"
            if param.grad.abs().sum() > 0:
                calib_has_grad = True

    assert calib_has_grad, "No gradient flow to calibrator parameters"

    # Check policy gradients
    policy_has_grad = False
    for param in pipeline.policy_net.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), "Policy grad contains NaN"
            assert not torch.isinf(param.grad).any(), "Policy grad contains Inf"
            if param.grad.abs().sum() > 0:
                policy_has_grad = True

    assert policy_has_grad, "No gradient flow to policy parameters"


def test_low_vega_gradient_gating():
    """
    Verify that low-vega regions do not trigger gradient explosion or division by zero.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = get_pipeline_setup(device)
    pipeline.train()

    # Use extremely short maturity and out-of-the-money strike to create low-vega condition
    market_surface = torch.full((2, 8, 11), 0.25, device=device, dtype=torch.float64)

    out = pipeline(
        market_surface=market_surface,
        N_steps=5,
        N_paths=20,
        S0=100.0,
        K=150.0,  # far OTM
        T=0.01,  # extremely short maturity
        r=0.0,
    )

    loss = out["loss_total"]
    loss.backward()

    # Verify that gradients are finite and did not explode
    for name, param in pipeline.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), (
                f"Gradient for {name} is not finite in low-vega region"
            )


def test_compliance_logging(caplog):
    """
    Verify that OOD parameters and PSI parameter drift trigger compliance logs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = get_pipeline_setup(device)

    # 1. Trigger OOD parameter warning
    # We pass out-of-bound parameters directly to check_and_clamp_ood
    # kappa=15.0 (max 10.0), rho=-1.5 (min -0.99)
    ood_params = torch.tensor([[15.0, 0.05, 0.3, -1.5, 0.05, 0.1]], device=device)

    with caplog.at_level(logging.WARNING):
        clamped = pipeline.check_and_clamp_ood(ood_params)

    # Check clamped values
    assert clamped[0, 0] <= 10.0
    assert clamped[0, 3] >= -0.99

    # Check captured logs
    ood_log_found = False
    for record in caplog.records:
        if "OOD_PARAMETER_WARNING" in record.message:
            ood_log_found = True
            break
    assert ood_log_found, "OOD parameter warning was not logged"

    # 2. Trigger PSI drift warning
    # We create a drifted distribution (shifted normal mean)
    caplog.clear()
    drifted_norm_params = torch.randn(100, 6, device=device) + 2.0  # shift mean by 2.0

    with caplog.at_level(logging.WARNING):
        pipeline.compute_psi(drifted_norm_params)

    drift_log_found = False
    for record in caplog.records:
        if "PARAMETER_DRIFT_WARNING" in record.message:
            drift_log_found = True
            break
    assert drift_log_found, "Parameter drift warning was not logged"
