import pytest
import numpy as np
import torch
from deepvol.arbitrage.projection_layer import DifferentiableArbitrageFreeProjection
from deepvol.mrm.arbitrage import check_arbitrage


def run_arbitrage_test_for_device_and_dtype(device: str, dtype: torch.dtype):
    # ── Setup grids ──
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    nT = len(T_grid)
    nK = len(K_grid)
    B = 2
    
    # Create an intentionally arbitrage-filled IV surface:
    # 1. Calendar violation: decrease IV at T=0.6 relative to T=0.3
    # 2. Butterfly violation: make a sharp dip in the middle strike (non-convex)
    iv_raw = np.full((B, nT, nK), 0.3, dtype=np.float64)
    iv_raw[:, 2, :] = 0.05  # calendar violation
    iv_raw[:, 0, 5] = 0.01  # butterfly violation
    
    iv_tensor = torch.tensor(iv_raw, dtype=dtype, device=device, requires_grad=True)
    projection = DifferentiableArbitrageFreeProjection(T_grid, K_grid, S0=1.0)
    projection.to(device)
    
    # Forward pass
    iv_projected = projection(iv_tensor)
    
    # Assert output shape and dtype matching input
    assert iv_projected.shape == (B, nT, nK)
    assert iv_projected.dtype == dtype
    
    # Convert to numpy for arbitrage validation
    iv_proj_np = iv_projected.detach().cpu().numpy()
    
    # Verify that all calendar and butterfly arbitrage violations are resolved
    for b in range(B):
        res = check_arbitrage(iv_proj_np[b], K_grid, T_grid, S=1.0)
        
        # Verify calendar arbitrage is completely resolved
        assert not res["calendar"]["has_arbitrage"], (
            f"Calendar arbitrage still exists on {device}/{dtype}: {res['calendar']['violations']}"
        )
        
        # Verify butterfly arbitrage is completely resolved
        assert not res["butterfly_durrleman"]["has_arbitrage"], f"Durrleman butterfly arbitrage still exists on {device}/{dtype}"
        assert not res["butterfly_price"]["has_arbitrage"], f"Price butterfly arbitrage still exists on {device}/{dtype}"
        
    # Verify gradient flow and differentiability
    loss = iv_projected.sum()
    loss.backward()
    
    assert iv_tensor.grad is not None
    assert torch.all(torch.isfinite(iv_tensor.grad))


def test_projection_layer_remediates_arbitrage_cpu_float32():
    run_arbitrage_test_for_device_and_dtype("cpu", torch.float32)


def test_projection_layer_remediates_arbitrage_cpu_float64():
    run_arbitrage_test_for_device_and_dtype("cpu", torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_projection_layer_remediates_arbitrage_cuda_float32():
    run_arbitrage_test_for_device_and_dtype("cuda", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_projection_layer_remediates_arbitrage_cuda_float64():
    run_arbitrage_test_for_device_and_dtype("cuda", torch.float64)


def test_projection_layer_differentiability_jacobian():
    # Test differentiability via computing the Jacobian of the output w.r.t the input
    T_grid = np.array([0.2, 0.5, 1.0])
    K_grid = np.array([-0.2, 0.0, 0.2])
    projection = DifferentiableArbitrageFreeProjection(T_grid, K_grid, S0=1.0)
    
    # Input tensor requiring gradients
    iv_raw = torch.tensor(
        [[[0.3, 0.2, 0.3],
          [0.3, 0.05, 0.3],
          [0.4, 0.4, 0.4]]],
        dtype=torch.float64,
        requires_grad=True
    )
    
    # Compute Jacobian matrix
    jac = torch.autograd.functional.jacobian(projection, iv_raw)
    
    # Assert shape (B, nT, nK, B, nT, nK)
    assert jac.shape == (1, 3, 3, 1, 3, 3)
    assert torch.all(torch.isfinite(jac))
