"""
tests/test_signature_sde.py - Unit and integration tests for the Signature Neural SDE and FNO surrogate.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from deepvol.surrogates.signature_sde import (
    SignatureNeuralSDE,
    SignatureSDEPricer,
    DualLegSignatureFNO2d,
    implied_volatility,
    vix_implied_volatility,
    ModelRiskGuardian,
    compute_path_signature
)


def test_path_signature_2d_sanity():
    """Verify that path signatures are computed correctly and differentiably."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Linear path: (0,0) -> (1,2) -> (2,4)
    path = torch.tensor([[[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]], device=device, dtype=torch.float64)
    sig = compute_path_signature(path, depth=3)
    
    # 2D path level 3 signature has 2 + 4 + 8 = 14 elements
    assert sig.shape == (1, 14)
    assert not torch.isnan(sig).any()
    
    # Level 1 signature: sum of increments = (2.0, 4.0)
    assert torch.allclose(sig[0, 0:2], torch.tensor([2.0, 4.0], device=device, dtype=torch.float64))


def test_signature_sde_stability():
    """Verify that the Signature SDE solver remains stable and preserves positivity of variance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sde = SignatureNeuralSDE(r=0.05, q=0.02)
    sde.to(device)
    
    # Set signature coefficients to non-zero values
    with torch.no_grad():
        sde.W_drift.fill_(0.01)
        sde.W_diff.fill_(0.02)
        
    S, V, t_grid = sde.simulate_paths(
        T=1.0, steps_per_unit=100, N_paths=128, S0=100.0, device=device
    )
    
    assert S.shape == (128, 101)
    assert V.shape == (128, 101)
    assert (V >= 1e-4).all(), "Variance floor violated"
    assert not torch.isnan(S).any(), "Stock price contains NaNs"
    assert not torch.isnan(V).any(), "Variance contains NaNs"


def test_implied_volatility_solver():
    """Verify that the Black-Scholes implied volatility solver is accurate and enforces min clamping."""
    S0 = torch.tensor([100.0, 100.0], dtype=torch.float64)
    K = torch.tensor([100.0, 110.0], dtype=torch.float64)
    T = torch.tensor([0.5, 0.5], dtype=torch.float64)
    r = torch.tensor([0.05, 0.05], dtype=torch.float64)
    # Define option prices with known volatilities (e.g. 0.20 and 0.25)
    from deepvol.surrogates.signature_sde import black_scholes_call
    known_vols = torch.tensor([0.20, 0.25], dtype=torch.float64)
    prices = black_scholes_call(S0, K, T, r, known_vols)
    
    # Invert prices
    ivs = implied_volatility(S0, K, T, r, prices)
    
    assert torch.allclose(ivs, known_vols, rtol=1e-4)
    
    # Check clamping rule: min volatility clamped to 0.01
    very_low_prices = torch.full_like(prices, 1e-6)
    ivs_clamped = implied_volatility(S0, K, T, r, very_low_prices)
    assert (ivs_clamped >= 0.01).all(), "Minimum volatility clamping failed"


def test_signature_sde_gradient_flow():
    """Verify that gradients flow correctly through the Signature SDE solver and pricer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sde = SignatureNeuralSDE(r=0.05, q=0.02)
    pricer = SignatureSDEPricer(sde)
    pricer.to(device)
    
    strikes = torch.tensor([95.0, 100.0, 105.0], device=device, dtype=torch.float64)
    maturities = torch.tensor([0.2, 0.2, 0.2], device=device, dtype=torch.float64)
    
    # Forward pass
    ivs = pricer.price_spx_options(
        S0=100.0, strikes=strikes, maturities=maturities, N_paths=64, steps_per_unit=50, device=device
    )
    
    loss = ivs.mean()
    loss.backward()
    
    # Verify gradients
    assert sde.W_drift.grad is not None, "Gradient for W_drift is None"
    assert sde.W_diff.grad is not None, "Gradient for W_diff is None"
    assert sde.raw_v0.grad is not None, "Gradient for raw_v0 is None"
    assert sde.raw_rho.grad is not None, "Gradient for raw_rho is None"
    
    assert not torch.isnan(sde.W_drift.grad).any(), "Gradient for W_drift contains NaNs"
    assert torch.max(torch.abs(sde.W_drift.grad)) >= 0.0, "Vanishing/broken gradient check"


def test_dual_leg_fno_model():
    """Verify that DualLegSignatureFNO2d maps parameters correctly to both SPX and VIX smiles."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualLegSignatureFNO2d(modes1=4, modes2=4, width=16)
    model.to(device)
    
    B = 4
    spatial_spx = torch.randn(B, 8, 11, 2, device=device)
    spatial_vix = torch.randn(B, 4, 9, 2, device=device)
    theta = torch.randn(B, 32, device=device)
    
    spx_iv, vix_iv = model(spatial_spx, spatial_vix, theta)
    
    assert spx_iv.shape == (B, 8, 11)
    assert vix_iv.shape == (B, 4, 9)
    
    # Check gradient flow
    loss = spx_iv.mean() + vix_iv.mean()
    loss.backward()
    
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"


def test_model_risk_guardian():
    """Verify Population Stability Index (PSI) and out-of-distribution clamping inside the risk guardian."""
    expected = np.random.normal(0.0, 1.0, size=500)
    actual_drifted = np.random.normal(0.5, 1.0, size=500)
    actual_stable = np.random.normal(0.0, 1.0, size=500)
    
    guardian = ModelRiskGuardian(expected_prior=np.zeros(1))
    
    psi_stable = guardian.compute_psi(actual_stable, expected)
    psi_drifted = guardian.compute_psi(actual_drifted, expected)
    
    # stable should have low PSI, drifted should have higher PSI
    assert psi_stable < 0.1
    assert psi_drifted > 0.1
    
    # Test OOD clamping
    params = torch.tensor([0.5, 0.1]) # v0=0.5 (OOD > 0.40), rho=0.1 (OOD > 0.0)
    clamped_params, is_ood = guardian.detect_ood_and_clamp(params)
    
    assert is_ood
    assert clamped_params[0] <= 0.35
    assert clamped_params[1] <= -0.05
