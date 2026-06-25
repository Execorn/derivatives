import pytest
import numpy as np
import torch
from deepvol.arbitrage.projection_layer import DifferentiableArbitrageFreeProjection, bs_call_price_pt
from deepvol.mrm.arbitrage import check_arbitrage


def test_projection_layer_remediates_arbitrage():
    # ── Setup grids ──
    T_grid = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
    K_grid = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    
    nT = len(T_grid)
    nK = len(K_grid)
    B = 2
    
    # ── Create an intentionally arbitrage-filled IV surface ──
    # We create a flat surface, then introduce manual violations:
    # 1. Calendar violation: decrease IV at T=0.6 relative to T=0.3
    # 2. Butterfly violation: make a sharp dip in the middle strike (non-convex)
    iv_raw = np.full((B, nT, nK), 0.3, dtype=np.float32)
    
    # Introduce calendar violation
    iv_raw[:, 2, :] = 0.1  # very low vol at T_idx=2 (T=0.6)
    
    # Introduce butterfly violation (non-convex strike shape)
    # At T_idx=0, make strikes K=0.0 have low vol, creating a concave valley
    iv_raw[:, 0, 5] = 0.05  # index 5 is strike 0.0
    
    # Convert to torch tensor with gradients enabled
    iv_tensor = torch.tensor(iv_raw, requires_grad=True)
    
    # Initialize projection layer
    projection = DifferentiableArbitrageFreeProjection(T_grid, K_grid, S0=1.0)
    
    # ── Forward pass ──
    iv_projected = projection(iv_tensor)
    
    # Check shape
    assert iv_projected.shape == (B, nT, nK)
    
    # Convert back to numpy for static checks
    iv_proj_np = iv_projected.detach().numpy()
    
    # Verify both batches are arbitrage-free
    for b in range(B):
        res = check_arbitrage(iv_proj_np[b], K_grid, T_grid, S=1.0)
        
        # Verify calendar arbitrage is completely resolved
        assert not res["calendar"]["has_arbitrage"], (
            f"Calendar arbitrage still exists: {res['calendar']['violations']}"
        )
        
        # Verify butterfly arbitrage is completely resolved
        assert not res["butterfly_durrleman"]["has_arbitrage"], "Butterfly (Durrleman) arbitrage still exists"
        assert not res["butterfly_price"]["has_arbitrage"], "Butterfly (Price) arbitrage still exists"
        
    # ── Check Gradient Flow ──
    # Compute simple loss and run backprop to verify differentiability
    loss = iv_projected.sum()
    loss.backward()
    
    assert iv_tensor.grad is not None
    assert torch.all(torch.isfinite(iv_tensor.grad))
    print("Differentiable projection checks completed successfully. All arbitrage resolved.")
