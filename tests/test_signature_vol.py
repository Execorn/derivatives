"""
tests/test_signature_vol.py - Unit and integration tests for the Signature Volatility Model.
"""

import pytest
import numpy as np
import torch
from src.pricing.signature_vol import (
    compute_path_signature,
    SignatureVolatilityModel,
    simulate_signature_vol_paths,
)


def test_pure_pytorch_signature_reproducibility():
    """Verify that our custom path signature matches expectations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create simple straight line path: (0, 0) -> (1, 2) -> (2, 4)
    # Increment delta = (1, 2) each step
    path = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]], device=device)
    
    sig = compute_path_signature(path, depth=2)
    # Level 1 should be sum of increments: (2.0, 4.0)
    assert torch.allclose(sig[0, 0:2], torch.tensor([2.0, 4.0], device=device))
    
    # Level 2 for piecewise straight line should be 0.5 * sum(deltas)^2
    # which is 0.5 * [2, 4] \otimes [2, 4] = [2, 4, 4, 8]
    expected_level2 = torch.tensor([2.0, 4.0, 4.0, 8.0], device=device)
    assert torch.allclose(sig[0, 2:6], expected_level2)


def test_signature_vol_positivity():
    """Verify that the simulator strictly enforces the positivity floor V_t >= 1e-4."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignatureVolatilityModel(device=device)
    
    # Set a very large negative coefficient for Level 1 W_t term to try and force negative variance
    with torch.no_grad():
        model.ell_raw[1] = -10.0
        
    S, V, V_raw, t_grid = model(
        T=1.0, steps_per_unit=100, N_paths=1000, positivity_func="relu"
    )
    
    assert (V >= 1e-4).all(), "Variance must be strictly greater than or equal to the floor (1e-4)"
    assert (V_raw < 0).any(), "Raw variance should have dropped below zero for the test case"


def test_signature_vol_martingale_property():
    """
    Verify that E[S_T] = S_0 (within Monte Carlo error bounds < 10 bps)
    under the risk-neutral measure with odd-order coefficients and rho < 0.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = SignatureVolatilityModel(device=device)
    
    # Configure model parameters
    with torch.no_grad():
        model.v0_raw.copy_(torch.tensor(np.log(0.04), device=device))  # v0 = 0.04
        model.rho_raw.copy_(torch.tensor(-0.5, device=device))         # rho < 0
        
        # Populate odd-order signature coefficients
        model.ell_raw[0] = 0.01   # level 1 time
        model.ell_raw[1] = -0.05  # level 1 W
        model.ell_raw[6] = 0.001  # level 3 time-time-time
        model.ell_raw[7] = -0.002 # level 3 time-time-W
        
    S0 = 100.0
    N_paths = 100000  # large count to minimize Monte Carlo sampling error
    
    # Perform forward pass
    S, V, _, _ = model(
        T=1.0, steps_per_unit=252, N_paths=N_paths, S0=S0, r=0.0, q=0.0, antithetic=True
    )
    
    # Calculate expectation and error
    E_ST = S[:, -1].mean().item()
    error_bps = abs(E_ST - S0) / S0 * 10000
    
    print(f"Expectation E[S_T]: {E_ST:.4f} (Target: {S0})")
    print(f"Martingale pricing error: {error_bps:.2f} bps")
    
    assert error_bps < 10.0, f"Martingale error of {error_bps:.2f} bps exceeds the 10 bps threshold"


def test_signature_vol_odd_order_masking():
    """Verify that even-order signature elements are strictly set to 0 in ell."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SignatureVolatilityModel(device=device)
    
    # Fill raw coefficients with ones
    with torch.no_grad():
        model.ell_raw.data.fill_(1.0)
        
    # Get constrained coefficients
    ell = model.get_constrained_ell()
    
    # Check even-order levels are 0
    even_indices = [2, 3, 4, 5] + list(range(14, 30))
    for idx in even_indices:
        assert ell[idx].item() == 0.0, f"Even-order element at index {idx} was not masked to 0"
        
    # Check odd-order levels are 1
    odd_indices = [0, 1] + list(range(6, 14))
    for idx in odd_indices:
        assert ell[idx].item() == 1.0, f"Odd-order element at index {idx} was changed"
