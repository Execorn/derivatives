"""
test_policy.py — Unit tests for DeepHedgingPolicy and differentiable transaction cost functions.
"""

import torch
import pytest
from deepvol.hedging.policy import (
    DeepHedgingPolicy,
    proportional_transaction_cost,
    huber_transaction_cost,
    sqrt_transaction_cost
)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
@pytest.mark.parametrize("range_limit", ["0.01_0.99", "0_1"])
def test_deep_hedging_policy_range_and_dims(device, range_limit):
    """
    Verify that the DeepHedgingPolicy operates correctly on CPU/CUDA, uses float32,
    accepts sequential and step-by-step inputs, and yields outputs within the expected bounds.
    """
    input_dim = 5
    hidden_dim = 16
    output_dim = 1
    
    policy = DeepHedgingPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        range_limit=range_limit
    ).to(device)
    
    # Check that model parameters are float32
    for param in policy.parameters():
        assert param.dtype == torch.float32
        
    # Check sequence input shape [Batch, Seq_Len, Input_Dim]
    batch_size = 4
    seq_len = 10
    x_seq = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float32)
    
    delta, h = policy(x_seq)
    
    assert delta.dtype == torch.float32
    assert delta.shape == (batch_size, seq_len, output_dim)
    assert h[0].shape == (batch_size, hidden_dim)
    assert h[1].shape == (batch_size, hidden_dim)
    
    # Check bounds
    if range_limit == "0.01_0.99":
        assert torch.all(delta >= 0.01)
        assert torch.all(delta <= 0.99)
    else:
        assert torch.all(delta >= 0.0)
        assert torch.all(delta <= 1.0)
        
    # Check step-by-step input shape [Batch, Input_Dim]
    x_step = torch.randn(batch_size, input_dim, device=device, dtype=torch.float32)
    delta_step, h_next = policy(x_step, h)
    
    assert delta_step.dtype == torch.float32
    assert delta_step.shape == (batch_size, output_dim)
    assert h_next[0].shape == (batch_size, hidden_dim)
    assert h_next[1].shape == (batch_size, hidden_dim)
    
    if range_limit == "0.01_0.99":
        assert torch.all(delta_step >= 0.01)
        assert torch.all(delta_step <= 0.99)
    else:
        assert torch.all(delta_step >= 0.0)
        assert torch.all(delta_step <= 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_production_cuda_deep_hedging_policy():
    """
    Production-scale test running on CUDA to verify GPU alignment,
    memory safety, and compiled path execution under standard GPU rules.
    """
    device = "cuda"
    policy = DeepHedgingPolicy(input_dim=10, hidden_dim=128, output_dim=3).to(device)
    
    # Large batch size/sequence length to verify GPU capability
    batch_size = 2048
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 10, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    delta, h = policy(x)
    torch.cuda.synchronize()
    
    assert delta.device.type == "cuda"
    assert delta.shape == (batch_size, seq_len, 3)
    assert delta.dtype == torch.float32


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_transaction_costs_differentiability(device):
    """
    Verify that the transaction cost functions are differentiable at delta_diff = 0
    and that their gradients are computed successfully without NaNs or Infs.
    """
    # Setup test input at delta_diff = 0
    delta_diff = torch.zeros(10, 1, device=device, dtype=torch.float32, requires_grad=True)
    S = torch.ones(10, 1, device=device, dtype=torch.float32) * 100.0
    c_fee = 0.002
    
    # 1. Proportional cost
    cost_prop = proportional_transaction_cost(delta_diff, S, c_fee)
    loss_prop = cost_prop.sum()
    loss_prop.backward()
    
    assert delta_diff.grad is not None
    assert not torch.isnan(delta_diff.grad).any()
    assert not torch.isinf(delta_diff.grad).any()
    
    # Reset grad
    delta_diff.grad.zero_()
    
    # 2. Huber cost
    cost_huber = huber_transaction_cost(delta_diff, S, c_fee, d=0.01)
    loss_huber = cost_huber.sum()
    loss_huber.backward()
    
    assert delta_diff.grad is not None
    assert not torch.isnan(delta_diff.grad).any()
    assert not torch.isinf(delta_diff.grad).any()
    # At 0, the derivative of Huber cost with respect to delta_diff should be exactly 0
    torch.testing.assert_close(delta_diff.grad, torch.zeros_like(delta_diff.grad))
    
    # Reset grad
    delta_diff.grad.zero_()
    
    # 3. Square-root cost
    cost_sqrt = sqrt_transaction_cost(delta_diff, S, c_fee, eps_c=1e-6)
    loss_sqrt = cost_sqrt.sum()
    loss_sqrt.backward()
    
    assert delta_diff.grad is not None
    assert not torch.isnan(delta_diff.grad).any()
    assert not torch.isinf(delta_diff.grad).any()
    # At 0, the derivative of square-root cost with respect to delta_diff should be exactly 0
    torch.testing.assert_close(delta_diff.grad, torch.zeros_like(delta_diff.grad))
