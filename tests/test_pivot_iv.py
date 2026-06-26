import pytest
import torch
from deepvol.hedging.pivot_iv import pivot_implied_vol, PIVOTImpliedVolFunction

def get_reference_price(sigma, S, K, T, r, q, is_call):
    # Reference BS pricing in float64 for generating test prices
    sigma = torch.as_tensor(sigma, dtype=torch.float64)
    S = torch.as_tensor(S, dtype=torch.float64)
    K = torch.as_tensor(K, dtype=torch.float64)
    T = torch.as_tensor(T, dtype=torch.float64)
    r = torch.as_tensor(r, dtype=torch.float64)
    q = torch.as_tensor(q, dtype=torch.float64)
    
    sqrt_T = torch.sqrt(T)
    denom = sigma * sqrt_T
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    
    SQRT_2 = 1.4142135623730951
    phi_d1 = 0.5 * (1.0 + torch.erf(d1 / SQRT_2))
    phi_d2 = 0.5 * (1.0 + torch.erf(d2 / SQRT_2))
    
    exp_q = torch.exp(-q * T)
    exp_r = torch.exp(-r * T)
    
    if is_call:
        return S * exp_q * phi_d1 - K * exp_r * phi_d2
    else:
        return K * exp_r * (1.0 - phi_d2) - S * exp_q * (1.0 - phi_d1)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_implied_vol_recovery(device):
    """
    Test that the solver recovers the correct implied volatility within MSE <= 1e-4.
    """
    # Create tensors of shape (10,)
    true_vol = torch.tensor([0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.5], device=device, dtype=torch.float32)
    S = torch.tensor([100.0] * 10, device=device, dtype=torch.float32)
    K = torch.tensor([90.0, 95.0, 100.0, 105.0, 110.0, 85.0, 100.0, 120.0, 70.0, 130.0], device=device, dtype=torch.float32)
    T = torch.tensor([0.25, 0.5, 1.0, 0.08, 1.5, 0.1, 2.0, 0.5, 3.0, 0.25], device=device, dtype=torch.float32)
    r = torch.tensor([0.05] * 10, device=device, dtype=torch.float32)
    q = torch.tensor([0.02] * 10, device=device, dtype=torch.float32)
    
    # Run for call and put options
    for is_call in [True, False]:
        # Compute reference price
        ref_prices = get_reference_price(true_vol.cpu(), S.cpu(), K.cpu(), T.cpu(), r.cpu(), q.cpu(), is_call).to(device=device, dtype=torch.float32)
        
        # Solve for implied volatility
        solved_vol = pivot_implied_vol(ref_prices, S, K, T, r, q, is_call=is_call)
        
        # Check dtype is preserved (input price was float32, output should be float32)
        assert solved_vol.dtype == torch.float32
        
        # Calculate MSE
        mse = torch.mean((solved_vol - true_vol) ** 2).item()
        print(f"Device: {device}, is_call: {is_call}, MSE: {mse:.2e}")
        assert mse <= 1e-4


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_vol_clamping(device):
    """
    Test that the solver correctly clamps low volatility to 0.01 and high volatility to 5.0.
    """
    S = torch.tensor([100.0, 100.0], device=device, dtype=torch.float32)
    K = torch.tensor([100.0, 100.0], device=device, dtype=torch.float32)
    T = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    r = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    q = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)
    
    # 1. Low vol: price is extremely low or below intrinsic (for call with S=K, intrinsic is 0, so price=0.0 is below min)
    # Price for vol = 0.001 is very small
    low_price = get_reference_price(0.001, 100.0, 100.0, 1.0, 0.0, 0.0, is_call=True).to(device)
    # Put extremely low price (even zero)
    zero_price = torch.tensor([0.0, low_price.item()], device=device, dtype=torch.float32)
    
    solved_low = pivot_implied_vol(zero_price, S, K, T, r, q, is_call=True)
    assert torch.allclose(solved_low, torch.tensor(0.01, device=device, dtype=torch.float32))
    
    # 2. High vol: price corresponds to vol = 6.0 (which is above 5.0)
    high_price_val = get_reference_price(6.0, 100.0, 100.0, 1.0, 0.0, 0.0, is_call=True).to(device)
    huge_price = torch.tensor([high_price_val.item(), 99.9], device=device, dtype=torch.float32)
    
    solved_high = pivot_implied_vol(huge_price, S, K, T, r, q, is_call=True)
    assert torch.allclose(solved_high, torch.tensor(5.0, device=device, dtype=torch.float32))


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_gradient_flow_and_gating(device):
    """
    Test that gradients flow correctly through the custom autograd function and
    do not overflow or produce NaNs for OTM options where vega is small.
    """
    # Extremely out of the money option (K is very far, maturity is short -> Vega is zero/near-zero)
    S = torch.tensor([100.0], device=device, dtype=torch.float32)
    K = torch.tensor([300.0], device=device, dtype=torch.float32)  # Deep OTM call
    T = torch.tensor([0.05], device=device, dtype=torch.float32)
    r = torch.tensor([0.05], device=device, dtype=torch.float32)
    q = torch.tensor([0.0], device=device, dtype=torch.float32)
    
    # Set price to 0.0 so it is clamped to min vol 0.01, giving Vega = 0.0 (below epsilon)
    price = torch.tensor([0.0], device=device, dtype=torch.float32, requires_grad=True)
    
    # Check forward
    solved_vol = pivot_implied_vol(price, S, K, T, r, q, is_call=True, vega_epsilon=1e-4)
    assert abs(solved_vol.item() - 0.01) < 1e-6  # Clamped to min
    
    # Backpropagate
    loss = solved_vol.sum()
    loss.backward()
    
    # Check that gradient is finite and not NaN
    assert price.grad is not None
    assert torch.isfinite(price.grad).all()
    # Gradient should be 1.0 / vega_epsilon because Vega is extremely small and gated by vega_epsilon
    expected_grad = 1.0 / 1e-4
    assert torch.allclose(price.grad, torch.tensor([expected_grad], device=device, dtype=torch.float32))


def test_gradcheck():
    """
    Perform a PyTorch gradcheck on the custom autograd function for moderate Vega options
    where low-vega gating is not active, to verify the mathematical correctness of the backward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use ATM option with moderate parameters to avoid gating
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.05], device=device, dtype=torch.float64)
    q = torch.tensor([0.02], device=device, dtype=torch.float64)
    
    # True vol of 0.4 gives a moderate price and high Vega
    true_vol = 0.4
    price_val = get_reference_price(true_vol, S, K, T, r, q, is_call=True).to(device)
    
    # Make price a parameter to check gradients
    price = price_val.clone().detach().requires_grad_(True)
    
    # Define a helper function for gradcheck (requires_grad only on price)
    def func(p):
        return PIVOTImpliedVolFunction.apply(p, S, K, T, r, q, True, 1e-4)
        
    # Run gradcheck
    assert torch.autograd.gradcheck(func, (price,), eps=1e-5, atol=1e-4, rtol=1e-4)
