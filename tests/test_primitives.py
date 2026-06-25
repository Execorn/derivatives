import pytest
import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import brentq
from deepvol.arbitrage.projection_layer import bs_call_price_pt, bs_iv_inversion_hybrid

# List of devices to test
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

# ─── SciPy Reference Functions ────────────────────────────────────────────────

def scipy_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """SciPy reference Black-Scholes call option pricing (r=0, q=0)."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)

def scipy_iv_inversion(price: float, S: float, K: float, T: float) -> float:
    """SciPy reference implied volatility solver using Brent's root finder."""
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic:
        return 0.01  # Clamped to minimum volatility in the model
    
    # Define objective function
    def obj(sigma):
        return scipy_call_price(S, K, T, sigma) - price
    
    # We want to find the root in [0.001, 10.0]
    # If the price is too close to intrinsic, or too high, we handle errors gracefully
    try:
        # Check boundary signs
        if obj(1e-6) * obj(10.0) > 0:
            # If no root in standard range, return clamped boundary
            if obj(1e-6) > 0:
                return 0.01
            else:
                return 5.0
        val = brentq(obj, 1e-6, 10.0, xtol=1e-15)
        return max(val, 0.01)
    except Exception:
        return 0.01

# ─── Unit Tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("device", DEVICES)
def test_bs_call_price_scipy_match(device):
    """
    Validate PyTorch vectorized Black-Scholes call option pricing
    against SciPy reference values under random perturbations.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 100
    # Generate random options parameters
    S_np = np.random.uniform(0.5, 2.0, num_samples)
    K_np = np.random.uniform(0.5, 2.0, num_samples)
    T_np = np.random.uniform(0.05, 3.0, num_samples)
    sigma_np = np.random.uniform(0.05, 2.0, num_samples)
    
    S_pt = torch.tensor(S_np, dtype=torch.float64, device=device)
    K_pt = torch.tensor(K_np, dtype=torch.float64, device=device)
    T_pt = torch.tensor(T_np, dtype=torch.float64, device=device)
    sigma_pt = torch.tensor(sigma_np, dtype=torch.float64, device=device)
    
    # Compute using PyTorch implementation
    prices_pt = bs_call_price_pt(S_pt, K_pt, T_pt, sigma_pt)
    prices_pt_cpu = prices_pt.cpu().numpy()
    
    # Compute using SciPy reference
    prices_scipy = np.array([
        scipy_call_price(S_np[i], K_np[i], T_np[i], sigma_np[i])
        for i in range(num_samples)
    ])
    
    # Verify matches to high precision (< 1e-10)
    abs_diff = np.abs(prices_pt_cpu - prices_scipy)
    max_diff = np.max(abs_diff)
    
    assert max_diff < 1e-10, f"Max difference {max_diff} exceeded tolerance 1e-10 on {device}"


@pytest.mark.parametrize("device", DEVICES)
def test_bs_iv_inversion_scipy_match(device):
    """
    Validate PyTorch vectorized implied volatility hybrid inversion solver
    against SciPy reference solver under random perturbations.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 100
    S_np = np.random.uniform(0.5, 2.0, num_samples)
    K_np = np.random.uniform(0.5, 2.0, num_samples)
    T_np = np.random.uniform(0.05, 3.0, num_samples)
    # Generate volatilities in [0.05, 2.0]
    sigma_true_np = np.random.uniform(0.05, 2.0, num_samples)
    
    # Compute option prices using SciPy first to get valid prices
    prices_np = np.array([
        scipy_call_price(S_np[i], K_np[i], T_np[i], sigma_true_np[i])
        for i in range(num_samples)
    ])
    
    S_pt = torch.tensor(S_np, dtype=torch.float64, device=device)
    K_pt = torch.tensor(K_np, dtype=torch.float64, device=device)
    T_pt = torch.tensor(T_np, dtype=torch.float64, device=device)
    prices_pt = torch.tensor(prices_np, dtype=torch.float64, device=device)
    
    # Recover volatilities using PyTorch hybrid solver
    sigma_recovered = bs_iv_inversion_hybrid(prices_pt, S_pt, K_pt, T_pt)
    sigma_recovered_cpu = sigma_recovered.cpu().numpy()
    
    # Recover volatilities using SciPy reference
    sigma_scipy = np.array([
        scipy_iv_inversion(prices_np[i], S_np[i], K_np[i], T_np[i])
        for i in range(num_samples)
    ])
    
    # Verify matches to < 1e-7 tolerance
    abs_diff = np.abs(sigma_recovered_cpu - sigma_scipy)
    max_diff = np.max(abs_diff)
    
    assert max_diff < 1e-7, f"Max difference {max_diff} exceeded tolerance 1e-7 on {device}"


@pytest.mark.parametrize("device", DEVICES)
def test_boundary_singularity_cases(device):
    """
    Test boundary/singularity cases:
    - T <= 1e-10
    - sigma <= 1e-10
    - zero volatility
    - extremely out-of-the-money or in-the-money options
    - zero time to maturity
    """
    # 1. Maturity Boundary T <= 1e-10 or sigma <= 1e-10
    S = torch.tensor([1.0, 1.2, 0.8], dtype=torch.float64, device=device)
    K = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, device=device)
    
    # Test T <= 1e-10
    T_zero = torch.tensor([0.0, 1e-11, 1e-10], dtype=torch.float64, device=device)
    sigma = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64, device=device)
    prices = bs_call_price_pt(S, K, T_zero, sigma)
    # Expected: max(S - K, 0.0)
    expected = torch.clamp(S - K, min=0.0)
    torch.testing.assert_close(prices, expected, atol=1e-15, rtol=1e-15)
    
    # Test sigma <= 1e-10
    T = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64, device=device)
    sigma_zero = torch.tensor([0.0, 1e-11, 1e-10], dtype=torch.float64, device=device)
    prices_zero_vol = bs_call_price_pt(S, K, T, sigma_zero)
    torch.testing.assert_close(prices_zero_vol, expected, atol=1e-15, rtol=1e-15)
    
    # 2. Implied Volatility Inversion Clamping to 0.01
    # If option price is equal to intrinsic value (or less), IV should be clamped to 0.01
    prices_intrinsic = torch.clamp(S - K, min=0.0)
    sigma_clamped = bs_iv_inversion_hybrid(prices_intrinsic, S, K, T)
    expected_clamped = torch.full_like(sigma_clamped, 0.01)
    torch.testing.assert_close(sigma_clamped, expected_clamped, atol=1e-15, rtol=1e-15)
    
    # 3. Extremely out-of-the-money options
    S_otm = torch.tensor([1.0], dtype=torch.float64, device=device)
    K_otm = torch.tensor([100.0], dtype=torch.float64, device=device)
    T_otm = torch.tensor([1.0], dtype=torch.float64, device=device)
    sigma_otm = torch.tensor([0.2], dtype=torch.float64, device=device)
    price_otm = bs_call_price_pt(S_otm, K_otm, T_otm, sigma_otm)
    
    # Should price to 0.0 or near it
    assert price_otm.item() >= 0.0
    assert price_otm.item() < 1e-30
    
    # Recovering vol from near zero price should result in clamping to 0.01
    sigma_otm_rec = bs_iv_inversion_hybrid(price_otm, S_otm, K_otm, T_otm)
    assert abs(sigma_otm_rec.item() - 0.01) < 1e-15
    
    # 4. Extremely in-the-money options
    S_itm = torch.tensor([100.0], dtype=torch.float64, device=device)
    K_itm = torch.tensor([1.0], dtype=torch.float64, device=device)
    T_itm = torch.tensor([1.0], dtype=torch.float64, device=device)
    sigma_itm = torch.tensor([0.2], dtype=torch.float64, device=device)
    price_itm = bs_call_price_pt(S_itm, K_itm, T_itm, sigma_itm)
    
    # Price should be extremely close to S - K = 99.0
    torch.testing.assert_close(price_itm, S_itm - K_itm, atol=1e-12, rtol=1e-12)
    
    # Recovering vol from near intrinsic price should result in clamping to 0.01
    sigma_itm_rec = bs_iv_inversion_hybrid(price_itm, S_itm, K_itm, T_itm)
    assert abs(sigma_itm_rec.item() - 0.01) < 1e-15


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_flow_and_double_precision(device):
    """
    Test double precision performance, backward pass, and gradient flow
    for both bs_call_price_pt and bs_iv_inversion_hybrid.
    """
    # Verify float64 dtype preservation
    S = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    K = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    T = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    sigma = torch.tensor([0.2], dtype=torch.float64, device=device, requires_grad=True)
    
    # 1. Price gradient flow
    price = bs_call_price_pt(S, K, T, sigma)
    assert price.dtype == torch.float64
    
    price.backward()
    
    # Verify gradients exist, are finite, and are correct
    assert S.grad is not None and torch.isfinite(S.grad)
    assert K.grad is not None and torch.isfinite(K.grad)
    assert T.grad is not None and torch.isfinite(T.grad)
    assert sigma.grad is not None and torch.isfinite(sigma.grad)
    
    # dC/dsigma is Vega. Vega for ATM 1y option is S * N'(d1) * sqrt(T).
    # d1 = 0.5 * 0.2 = 0.1. N'(0.1) = exp(-0.005)/sqrt(2pi) = 0.39695
    # Vega = 1.0 * 0.39695 * 1.0 = 0.39695
    expected_vega = 0.3969525474770118
    torch.testing.assert_close(sigma.grad, torch.tensor([expected_vega], dtype=torch.float64, device=device), atol=1e-5, rtol=1e-5)
    
    # 2. Inversion solver gradient flow
    # Reset gradients
    S = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    K = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    T = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)
    
    # Create target price that corresponds to sigma = 0.25
    target_sigma = torch.tensor([0.25], dtype=torch.float64, device=device)
    target_price = bs_call_price_pt(S.detach(), K.detach(), T.detach(), target_sigma)
    target_price.requires_grad = True
    
    sigma_rec = bs_iv_inversion_hybrid(target_price, S, K, T)
    assert sigma_rec.dtype == torch.float64
    torch.testing.assert_close(sigma_rec, target_sigma, atol=1e-6, rtol=1e-6)
    
    loss = sigma_rec.sum()
    loss.backward()
    
    # Verify gradients propagate to inputs
    assert target_price.grad is not None and torch.isfinite(target_price.grad)
    assert S.grad is not None and torch.isfinite(S.grad)
    assert K.grad is not None and torch.isfinite(K.grad)
    assert T.grad is not None and torch.isfinite(T.grad)
    
    # Gradient of sigma with respect to price should be 1 / Vega.
    # For sigma = 0.25: d1 = 0.5 * 0.25 = 0.125. N'(0.125) = exp(-0.0078125)/sqrt(2pi) = 0.395788
    # Vega = 0.395788
    # 1/Vega = 2.5266
    expected_grad_price = 1.0 / (1.0 * (np.exp(-0.5 * 0.125**2) / np.sqrt(2 * np.pi)) * 1.0)
    torch.testing.assert_close(target_price.grad, torch.tensor([expected_grad_price], dtype=torch.float64, device=device), atol=1e-5, rtol=1e-5)
    
    # 3. Check that gradient with respect to price is positive
    assert target_price.grad.item() > 0.0
