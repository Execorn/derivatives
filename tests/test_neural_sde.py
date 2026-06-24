import torch
import pytest
import torchsde
from deepvol.models.neural_sde import NeuralSDE, NeuralSDEPricer, compute_calibration_loss

def test_neural_sde_vol_positivity():
    """
    Volatility positivity check: verify that V_t > 0 for 1,000 simulated paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Neural SDE and Pricer
    sde = NeuralSDE(r=0.05, q=0.02, rho_init=-0.7, hidden_dim=16, epsilon=1e-4)
    pricer = NeuralSDEPricer(sde, v0_init=0.04)
    pricer.to(device)
    
    # Inputs for pricing
    S0 = 100.0
    strikes = torch.tensor([90.0, 100.0, 110.0], device=device)
    maturities = torch.tensor([0.1, 0.2, 0.3], device=device)
    
    # Run simulation
    N_paths = 1000
    prices, ys = pricer.price_options(
        S0=S0,
        strikes=strikes,
        maturities=maturities,
        N_paths=N_paths,
        dt=0.01,
        method="euler"
    )
    
    # Extract variance paths (V_t)
    # ys shape: (N_ts, N_paths, 2)
    # V_t is ys[:, :, 1]
    v_t = ys[:, :, 1]
    
    # Assertions
    assert v_t.shape == (4, N_paths)  # t=0, t=0.1, t=0.2, t=0.3
    assert (v_t >= 1e-4).all(), "Variance contains values below the simulation floor 1e-4"
    assert not torch.isnan(v_t).any(), "Variance contains NaN values"
    assert not torch.isinf(v_t).any(), "Variance contains infinite values"
    assert (prices > 0).all(), "Option prices should be positive"


def test_neural_sde_differentiability():
    """
    Differentiability check: verify that torch.autograd.grad returns valid,
    non-zero gradients for drift and diffusion parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Neural SDE and Pricer
    sde = NeuralSDE(r=0.05, q=0.02, rho_init=-0.7, hidden_dim=16, epsilon=1e-4)
    pricer = NeuralSDEPricer(sde, v0_init=0.04)
    pricer.to(device)
    
    # Setup some target/market prices and vegas for calibration loss
    S0 = 100.0
    strikes = torch.tensor([95.0, 100.0, 105.0], device=device)
    maturities = torch.tensor([0.2, 0.2, 0.2], device=device)
    market_prices = torch.tensor([8.0, 4.5, 2.0], device=device)
    vegas = torch.tensor([0.2, 0.2, 0.2], device=device)
    
    # Ensure gradients can be tracked
    # We want to check gradients for:
    # 1. pricer.raw_v0
    # 2. sde.raw_rho
    # 3. sde.drift_mlp parameters
    # 4. sde.diff_mlp parameters
    assert pricer.raw_v0.requires_grad
    assert sde.raw_rho.requires_grad
    
    # Run the model forward
    N_paths = 128  # Keep paths small for fast gradient check
    prices, ys = pricer.price_options(
        S0=S0,
        strikes=strikes,
        maturities=maturities,
        N_paths=N_paths,
        dt=0.01,
        method="euler"
    )
    
    # Compute loss
    loss_dict = compute_calibration_loss(
        model_prices=prices,
        market_prices=market_prices,
        vegas=vegas,
        ys=ys,
        lambda_bound=0.01,
        epsilon=1e-4
    )
    loss = loss_dict["loss"]
    
    # Get parameters we want gradients for
    params_to_check = {
        "raw_v0": pricer.raw_v0,
        "raw_rho": sde.raw_rho,
    }
    
    # Add drift MLP parameters
    drift_params = list(sde.drift_mlp.parameters())
    assert len(drift_params) > 0
    params_to_check["drift_mlp_weight"] = drift_params[0]
    
    # Add diff MLP parameters
    diff_params = list(sde.diff_mlp.parameters())
    assert len(diff_params) > 0
    params_to_check["diff_mlp_weight"] = diff_params[0]
    
    # Compute gradients using autograd
    grads = torch.autograd.grad(
        outputs=loss,
        inputs=list(params_to_check.values()),
        retain_graph=True,
        allow_unused=False
    )
    
    # Verify that gradients are not None, not NaN, and non-zero
    for name, grad in zip(params_to_check.keys(), grads):
        assert grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(grad).any(), f"Gradient for {name} contains NaNs"
        assert not torch.isinf(grad).any(), f"Gradient for {name} contains Infs"
        assert torch.max(torch.abs(grad)) > 1e-8, f"Gradient for {name} is zero/vanishing: {grad}"
        print(f"Gradient for {name} verified successfully: max abs value = {torch.max(torch.abs(grad)).item()}")
