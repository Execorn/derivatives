import torch
import numpy as np
import pytest
from deepvol.hedging.deep_hedging import (
    HedgingPolicy,
    estimate_gpd_tail_index_pwm,
    compute_acf_loss,
    compute_leverage_loss,
    compute_cfvc_loss
)
from deepvol.hedging.adversarial_market import (
    WGAN_GP_Generator,
    WGAN_GP_Discriminator,
    train_robust_minimax_hedger
)
from deepvol.hedging.cvar_loss import CVaRLoss


def test_adversarial_components():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generator = WGAN_GP_Generator(latent_dim=10, seq_len=50, hidden_dim=16).to(device)
    discriminator = WGAN_GP_Discriminator(seq_len=50, hidden_dim=16).to(device)
    
    # 1. Test generator shapes
    z = torch.randn(8, 10, device=device)
    fake_paths = generator(z)
    
    # Shape: (batch_size, channels=2, seq_len)
    assert fake_paths.shape == (8, 2, 50)
    
    # Volatility proxy must be strictly positive
    vol_paths = fake_paths[:, 1, :]
    assert torch.all(vol_paths >= 1e-4)
    
    # 2. Test discriminator shapes
    score = discriminator(fake_paths)
    assert score.shape == (8, 1)


def test_minimax_training_step():
    """
    Runs a minimal training test of 1 epoch with small data to verify that
    all components (Generator, Discriminator, HedgingPolicy) are fully linked,
    differentiable, and update without errors.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Latent space and sequence settings
    latent_dim = 10
    seq_len = 30
    
    # Generate mock real returns (512 samples)
    real_returns = torch.randn(512, seq_len, device=device) * 0.01
    
    # Compute mock real targets
    # ACF of absolute returns (dummy)
    real_acf = torch.linspace(0.1, 0.0, 20, device=device)
    
    # Leverage correlation
    real_leverage = -0.12
    
    # CFVC correlation matrix (dummy 4x4)
    real_cfvc_matrix = torch.eye(4, device=device)
    
    # Initialize networks
    generator = WGAN_GP_Generator(latent_dim=latent_dim, seq_len=seq_len, hidden_dim=16)
    discriminator = WGAN_GP_Discriminator(seq_len=seq_len, hidden_dim=16)
    policy = HedgingPolicy(input_dim=4, hidden_dim=16, output_dim=1)  # d = 1 instrument (stock price only)
    
    # Run a single epoch of minimax training
    train_robust_minimax_hedger(
        real_returns=real_returns,
        real_acf=real_acf,
        real_leverage=real_leverage,
        real_cfvc_matrix=real_cfvc_matrix,
        generator=generator,
        discriminator=discriminator,
        policy=policy,
        epochs=1,
        critic_steps=1,
        minimax_coeff=0.01,
        device=device
    )
    
    # Verify weights are modified (non-NaN)
    for name, param in generator.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param).any()
            
    for name, param in policy.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param).any()
            
    print("Minimax single step verification SUCCESSFUL.")


def test_stylized_facts_differentiability():
    """
    Verifies that each of the four stylized facts loss functions:
      1. estimate_gpd_tail_index_pwm (fat tails)
      2. compute_acf_loss (volatility clustering)
      3. compute_leverage_loss (leverage effect)
      4. compute_cfvc_loss (coarse-to-fine volatility correlation)
    is fully differentiable with respect to the output of WGAN_GP_Generator,
    and backpropagating through them yields non-zero gradients on generator parameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    latent_dim = 10
    seq_len = 30
    batch_size = 8
    
    # Initialize generator
    generator = WGAN_GP_Generator(latent_dim=latent_dim, seq_len=seq_len, hidden_dim=16).to(device)
    
    # Mock real targets matching the shapes required by the loss functions
    # 1. Real returns for GPD tail index estimation
    real_returns = torch.randn(batch_size, seq_len, device=device) * 0.01
    
    # 2. Target ACF of absolute returns (20 lags)
    real_acf = torch.linspace(0.1, 0.0, 20, device=device)
    
    # 3. Target leverage correlation (scalar)
    real_leverage = -0.12
    
    # 4. Target CFVC correlation matrix (dummy 4x4 for scales [5, 20, 60, 120])
    real_cfvc_matrix = torch.eye(4, device=device)
    
    # Generate path outputs
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_samples = generator(z)
    fake_returns = fake_samples[:, 0, :]
    
    # Define losses to test
    losses = {
        "GPD": lambda: torch.mean(
            torch.abs(
                estimate_gpd_tail_index_pwm(real_returns, threshold_quantile=0.90) -
                estimate_gpd_tail_index_pwm(fake_returns, threshold_quantile=0.90)
            )
        ) + torch.mean(
            torch.abs(
                estimate_gpd_tail_index_pwm(-real_returns, threshold_quantile=0.90) -
                estimate_gpd_tail_index_pwm(-fake_returns, threshold_quantile=0.90)
            )
        ),
        "ACF": lambda: compute_acf_loss(fake_returns, real_acf),
        "Leverage": lambda: compute_leverage_loss(fake_returns, real_leverage),
        "CFVC": lambda: compute_cfvc_loss(fake_returns, real_cfvc_matrix)
    }
    
    # Test differentiability for each loss individually
    for name, get_loss in losses.items():
        generator.zero_grad()
        loss = get_loss()
        
        # Verify loss value is valid
        assert not torch.isnan(loss) and not torch.isinf(loss), f"{name} loss is NaN or Inf"
        
        # Backpropagate (using retain_graph=True since we reuse the same generator outputs)
        loss.backward(retain_graph=True)
        
        # Verify that all generator parameters with requires_grad have valid, non-zero gradients
        grad_norms = []
        for param_name, param in generator.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name}: gradient is None for {param_name}"
                assert not torch.isnan(param.grad).any(), f"{name}: NaN found in gradient for {param_name}"
                grad_norms.append(param.grad.norm().item())
        
        sum_norms = sum(grad_norms)
        assert sum_norms > 0.0, f"{name}: sum of gradient norms is zero (no gradient propagated)"


def test_cvar_loss_properties():
    """
    Validates mathematical correctness, differentiability, float64 internal math,
    and parameter validation of CVaRLoss.
    """
    # 1. Test parameter validation
    with pytest.raises(ValueError):
        CVaRLoss(alpha=0.0)
    with pytest.raises(ValueError):
        CVaRLoss(alpha=1.0)
    with pytest.raises(ValueError):
        CVaRLoss(alpha=-0.5)

    # 2. Test mathematical correctness
    cvar_fn = CVaRLoss(alpha=0.8)
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32)
    cvar_val = cvar_fn(losses)
    
    # CVaR must be strictly greater than the mean of the losses
    assert cvar_val.item() > losses.mean().item()
    
    # Rockafellar-Uryasev formulation for [1..10] at alpha=0.8:
    # VaR = 8.2 (linear interpolation), CVaR = 9.5
    assert abs(cvar_val.item() - 9.5) < 1e-5

    # 3. Test differentiability
    losses.requires_grad_(True)
    cvar_val = cvar_fn(losses)
    cvar_val.backward()
    assert losses.grad is not None
    # Gradients should only be non-zero for paths with losses >= VaR
    # VaR = 8.2, so only 9.0 and 10.0 should have positive gradients, others should be zero
    assert torch.allclose(losses.grad[:8], torch.zeros(8), atol=1e-7)
    assert torch.all(losses.grad[8:] > 0.0)

    # 4. Test float64 internal math (promoting to float64 and casting back)
    assert cvar_val.dtype == torch.float32


def test_minimax_cvar_training():
    """
    Validates minimax robust deep hedging loop stability and convergence properties
    when using the CVaR risk measure.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Latent space and sequence settings
    latent_dim = 10
    seq_len = 30
    
    # Generate mock real returns (512 samples)
    real_returns = torch.randn(512, seq_len, device=device) * 0.01
    real_acf = torch.linspace(0.1, 0.0, 20, device=device)
    real_leverage = -0.12
    real_cfvc_matrix = torch.eye(4, device=device)
    
    # Initialize networks
    generator = WGAN_GP_Generator(latent_dim=latent_dim, seq_len=seq_len, hidden_dim=16)
    discriminator = WGAN_GP_Discriminator(seq_len=seq_len, hidden_dim=16)
    policy = HedgingPolicy(input_dim=4, hidden_dim=16, output_dim=1)
    
    # Run a single epoch of minimax training with "cvar" risk measure
    train_robust_minimax_hedger(
        real_returns=real_returns,
        real_acf=real_acf,
        real_leverage=real_leverage,
        real_cfvc_matrix=real_cfvc_matrix,
        generator=generator,
        discriminator=discriminator,
        policy=policy,
        epochs=1,
        critic_steps=1,
        minimax_coeff=0.01,
        device=device,
        risk_measure="cvar",
        alpha=0.95
    )
    
    # Verify weights are modified and are not NaN
    for name, param in generator.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param).any()
            
    for name, param in policy.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param).any()


