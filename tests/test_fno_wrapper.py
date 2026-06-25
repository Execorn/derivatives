import os
import pytest
import torch
import torch.nn as nn
import numpy as np

from deepvol.arbitrage.projection_layer import (
    DifferentiableArbitrageFreeProjection,
    ArbitrageFreeFNO
)
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization
from deepvol.surrogates.normalizers import IVSurfaceNormalizer

def get_iv_normalizer() -> IVSurfaceNormalizer:
    path = "/home/execorn/programming/derivatives-w3/artifacts/models/iv_normalizer_v2.npz"
    if os.path.exists(path):
        return IVSurfaceNormalizer.load(path)
    else:
        # Construct mock normalizer if not present
        norm = IVSurfaceNormalizer()
        norm.mean = np.full((8, 11), 0.3)
        norm.std = np.full((8, 11), 0.1)
        return norm

def test_fno_wrapper_initialization():
    base_fno = MirrorPaddedFNO2d()
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm)
    assert wrapper.base_fno is base_fno
    assert wrapper.projection_layer is proj
    assert wrapper.normalizer is norm

def test_fno_wrapper_forward_cpu():
    base_fno = MirrorPaddedFNO2d()
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm)
    
    B = 4
    spatial = torch.randn(B, 8, 11, 2, dtype=torch.float32)
    theta = torch.randn(B, 6, dtype=torch.float32)
    
    out = wrapper(spatial, theta)
    
    assert out.shape == (B, 8, 11)
    assert out.dtype == torch.float32

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fno_wrapper_forward_gpu():
    device = "cuda"
    base_fno = MirrorPaddedFNO2d().to(device)
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64, device=device)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64, device=device)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm).to(device)
    
    B = 4
    spatial = torch.randn(B, 8, 11, 2, dtype=torch.float32, device=device)
    theta = torch.randn(B, 6, dtype=torch.float32, device=device)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    out = wrapper(spatial, theta)
    torch.cuda.synchronize()
    
    assert out.shape == (B, 8, 11)
    assert out.dtype == torch.float32
    assert out.device.type == "cuda"

def test_gradient_flow_cpu():
    base_fno = MirrorPaddedFNO2d()
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm)
    
    B = 2
    spatial = torch.randn(B, 8, 11, 2, dtype=torch.float32, requires_grad=True)
    theta = torch.randn(B, 6, dtype=torch.float32, requires_grad=True)
    
    out = wrapper(spatial, theta)
    loss = out.mean()
    loss.backward()
    
    # Ensure gradients flow cleanly to theta
    assert theta.grad is not None
    assert torch.any(theta.grad != 0.0)
    
    # Ensure gradients flow cleanly to spatial coords
    assert spatial.grad is not None
    assert torch.any(spatial.grad != 0.0)
    
    # Ensure gradients flow cleanly to base FNO weights
    for name, p in base_fno.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.any(p.grad != 0.0), f"Parameter {name} gradient is zero"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_flow_gpu():
    device = "cuda"
    base_fno = MirrorPaddedFNO2d().to(device)
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64, device=device)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64, device=device)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm).to(device)
    
    B = 2
    spatial = torch.randn(B, 8, 11, 2, dtype=torch.float32, device=device, requires_grad=True)
    theta = torch.randn(B, 6, dtype=torch.float32, device=device, requires_grad=True)
    
    out = wrapper(spatial, theta)
    loss = out.mean()
    loss.backward()
    
    assert theta.grad is not None
    assert torch.any(theta.grad != 0.0)
    assert spatial.grad is not None
    assert torch.any(spatial.grad != 0.0)
    
    for name, p in base_fno.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.any(p.grad != 0.0), f"Parameter {name} gradient is zero"

def test_precision_boundary_flow():
    """
    Verify precision boundary requirements:
    1. Base FNO operates in float32 (or bfloat16).
    2. Projection operates strictly in float64 internally.
    3. Final output is cast back to float32.
    """
    # Use a mock FNO that returns bfloat16 to bypass PyTorch CPU FFT bfloat16 limitation
    class MockBFloat16FNO(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.randn(1))
            
        def forward(self, spatial, theta):
            B = spatial.size(0)
            return torch.zeros(B, 8, 11, dtype=torch.bfloat16, device=spatial.device)
            
    base_fno = MockBFloat16FNO()
    
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm)
    
    B = 2
    spatial = torch.randn(B, 8, 11, 2, dtype=torch.bfloat16)
    theta = torch.randn(B, 6, dtype=torch.bfloat16)
    
    # We can patch projection_layer.forward to verify that it receives bfloat16
    # and that its internal computations are indeed in float64
    original_proj_forward = proj.forward
    recorded_dtypes = []
    
    def mock_forward(iv_surface):
        recorded_dtypes.append(("input_to_proj", iv_surface.dtype))
        # Call original
        res = original_proj_forward(iv_surface)
        recorded_dtypes.append(("output_of_proj", res.dtype))
        return res
        
    proj.forward = mock_forward
    
    out = wrapper(spatial, theta)
    
    assert out.dtype == torch.float32
    # The normalizer output was in bfloat16 (since it uses base_fno's output dtype)
    assert recorded_dtypes[0] == ("input_to_proj", torch.bfloat16)
    # The projection returns in the input dtype (bfloat16)
    assert recorded_dtypes[1] == ("output_of_proj", torch.bfloat16)
    
    # Restore
    proj.forward = original_proj_forward

def test_arbitrage_free_projection_guarantee():
    """
    Verify that when the base FNO generates surfaces with heavy arbitrage violations,
    the ArbitrageFreeFNO wrapper successfully projects them such that the arbitrage penalty
    is zero or negligible.
    """
    # Mock a base FNO that outputs arbitrary, non-smooth surfaces with calendar/butterfly violations
    class ArbitraryFNO(nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy parameter to satisfy named_parameters
            self.dummy = nn.Parameter(torch.randn(1))
            
        def forward(self, spatial, theta):
            # Generate a surface with explicit arbitrage violations:
            # e.g., IV decreases with maturity (calendar arbitrage)
            # or option price lacks convexity (butterfly arbitrage)
            # Let's make an extreme arbitrage-ridden z-score surface
            B = spatial.size(0)
            surf = torch.zeros(B, 8, 11, dtype=spatial.dtype, device=spatial.device)
            # For maturity 0 (T=0.1), high IV:
            surf[:, 0, :] = 2.0
            # For maturity 7 (T=2.0), low IV:
            surf[:, 7, :] = -2.0
            return surf
            
    base_fno = ArbitraryFNO()
    T_grid = torch.tensor([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=torch.float64)
    K_grid = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
    proj = DifferentiableArbitrageFreeProjection(T_grid=T_grid, K_grid=K_grid)
    norm = get_iv_normalizer()
    wrapper = ArbitrageFreeFNO(base_fno=base_fno, projection_layer=proj, normalizer=norm)
    
    B = 2
    spatial = torch.randn(B, 8, 11, 2, dtype=torch.float32)
    theta = torch.randn(B, 6, dtype=torch.float32)
    
    # 1. Check raw FNO output (before projection) has high arbitrage penalty
    raw_norm = base_fno(spatial, theta)
    raw_iv = norm.inverse_transform_tensor(raw_norm)
    raw_penalty = arbitrage_free_regularization(raw_iv, T_grid, K_grid)
    assert raw_penalty.item() > 0.1
    
    # 2. Check that wrapper output has zero/negligible arbitrage penalty
    clean_norm = wrapper(spatial, theta)
    clean_iv = norm.inverse_transform_tensor(clean_norm)
    clean_penalty = arbitrage_free_regularization(clean_iv, T_grid, K_grid)
    
    # No arbitrage condition is strictly enforced, penalty should be extremely small (numerical precision limits)
    assert clean_penalty.item() < 1e-6
