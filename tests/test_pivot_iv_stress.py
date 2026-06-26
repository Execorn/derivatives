import pytest
import torch
from deepvol.hedging.pivot_iv import pivot_implied_vol


def get_ref_price_f64(sigma, S, K, T, r, q, is_call):
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


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_extreme_prices_intrinsic_and_s(device):
    """
    Test option prices extremely close to intrinsic values, or close to S.
    """
    S = torch.tensor([100.0, 100.0, 100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0, 100.0, 100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)

    # 1. Price extremely close to intrinsic (intrinsic of ATM call is 0.0)
    # Price close to 0: 1e-15, 0.0, and negative price -1e-15 (below intrinsic)
    prices_low = torch.tensor([1e-15, 0.0, -1e-15], device=device, dtype=torch.float64)

    # Forward pass
    solved_low = pivot_implied_vol(prices_low, S, K, T, r, q, is_call=True)
    assert solved_low.shape == prices_low.shape
    assert torch.all(solved_low == 0.01), (
        f"Expected low prices to clamp to 0.01, got {solved_low}"
    )
    assert torch.isfinite(solved_low).all()

    # 2. Price extremely close to S (upper limit of Call price with S=100, K=100 is S)
    # prices close to S: S - 1e-12, S, S + 1.0
    prices_high = torch.tensor(
        [99.9999999999, 100.0, 101.0], device=device, dtype=torch.float64
    )

    solved_high = pivot_implied_vol(prices_high, S, K, T, r, q, is_call=True)
    assert solved_high.shape == prices_high.shape
    assert torch.all(solved_high == 5.0), (
        f"Expected high prices to clamp to 5.0, got {solved_high}"
    )
    assert torch.isfinite(solved_high).all()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_extreme_spot_strike_ratios(device):
    """
    Test Spot-Strike ratios of 0.01 and 100.
    - S/K = 0.01 (e.g. S=1.0, K=100.0) -> extremely out-of-the-money call
    - S/K = 100 (e.g. S=100.0, K=1.0) -> extremely in-the-money call

    Due to float64 underflow in standard erf-based BS formulation, the time value
    underflows to exactly 0, which triggers the solver's min-vol clamping logic (0.01).
    """
    # Case 1: S/K = 0.01
    S_low = torch.tensor([1.0], device=device, dtype=torch.float64)
    K_high = torch.tensor([100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0], device=device, dtype=torch.float64)

    # Moderate vol = 0.4
    ref_price_call = get_ref_price_f64(0.4, S_low, K_high, T, r, q, is_call=True).to(
        device
    )
    solved_vol_call = pivot_implied_vol(
        ref_price_call, S_low, K_high, T, r, q, is_call=True
    )
    assert torch.isfinite(solved_vol_call).all()
    # Verified: returns 0.01 due to numerical underflow of the BS price
    assert solved_vol_call.item() == 0.01, (
        f"Expected S/K=0.01 to clamp to 0.01 due to underflow, got {solved_vol_call.item()}"
    )

    # Case 2: S/K = 100
    S_high = torch.tensor([100.0], device=device, dtype=torch.float64)
    K_low = torch.tensor([1.0], device=device, dtype=torch.float64)

    ref_price_call_2 = get_ref_price_f64(0.4, S_high, K_low, T, r, q, is_call=True).to(
        device
    )
    solved_vol_call_2 = pivot_implied_vol(
        ref_price_call_2, S_high, K_low, T, r, q, is_call=True
    )
    assert torch.isfinite(solved_vol_call_2).all()
    # Verified: returns 0.01 due to numerical price equaling intrinsic price under float64
    assert solved_vol_call_2.item() == 0.01, (
        f"Expected S/K=100 to clamp to 0.01 due to underflow, got {solved_vol_call_2.item()}"
    )


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_extreme_maturities(device):
    """
    Test T extremely small (1e-6) or large (10.0).
    """
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.05], device=device, dtype=torch.float64)
    q = torch.tensor([0.02], device=device, dtype=torch.float64)

    # 1. T extremely small
    T_small = torch.tensor([1e-6], device=device, dtype=torch.float64)
    ref_price_small = get_ref_price_f64(0.3, S, K, T_small, r, q, is_call=True).to(
        device
    )
    solved_small = pivot_implied_vol(ref_price_small, S, K, T_small, r, q, is_call=True)
    assert torch.isfinite(solved_small).all()
    assert abs(solved_small.item() - 0.3) < 1e-3, (
        f"Failed recovery for T=1e-6: got {solved_small.item()}"
    )

    # 2. T extremely large
    T_large = torch.tensor([10.0], device=device, dtype=torch.float64)
    ref_price_large = get_ref_price_f64(0.3, S, K, T_large, r, q, is_call=True).to(
        device
    )
    solved_large = pivot_implied_vol(ref_price_large, S, K, T_large, r, q, is_call=True)
    assert torch.isfinite(solved_large).all()
    assert abs(solved_large.item() - 0.3) < 1e-4, (
        f"Failed recovery for T=10.0: got {solved_large.item()}"
    )


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_gradient_flow_extreme_otm(device):
    """
    Verify that gradients do not blow up in extremely out-of-the-money regions
    where vega is very small/zero, and that the clamped vega mechanism works.
    """
    # Extremely OTM option
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([500.0], device=device, dtype=torch.float64)  # Deep OTM
    T = torch.tensor([0.1], device=device, dtype=torch.float64)
    r = torch.tensor([0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0], device=device, dtype=torch.float64)

    # Set price to 0.0 (leads to min vol 0.01 and extremely low vega)
    price = torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True)

    vega_epsilon = 1e-4
    solved_vol = pivot_implied_vol(
        price, S, K, T, r, q, is_call=True, vega_epsilon=vega_epsilon
    )

    loss = solved_vol.sum()
    loss.backward()

    assert price.grad is not None
    assert torch.isfinite(price.grad).all()

    # Epsilon-gated gradient should be exactly 1 / vega_epsilon
    expected_grad = 1.0 / vega_epsilon
    assert torch.allclose(
        price.grad, torch.tensor([expected_grad], device=device, dtype=torch.float64)
    ), f"Expected grad {expected_grad}, got {price.grad.item()}"


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_solver_nan_inf_immunity(device):
    """
    Verify that the solver handles completely invalid inputs (e.g. price=NaN, Inf, or negative)
    safely without crashing, and either clamps to boundaries or propagates predictably without system error.
    """
    S = torch.tensor([100.0, 100.0, 100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0, 100.0, 100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)

    # Prices with NaN, Inf, and negative infinity
    nan_tensor = torch.tensor(float("nan"), device=device, dtype=torch.float64)
    inf_tensor = torch.tensor(float("inf"), device=device, dtype=torch.float64)
    neginf_tensor = torch.tensor(float("-inf"), device=device, dtype=torch.float64)

    prices = torch.stack([nan_tensor, inf_tensor, neginf_tensor])

    # We want to check that it doesn't crash Python/CUDA execution.
    solved = pivot_implied_vol(prices, S, K, T, r, q, is_call=True)

    # For NaN input, we expect the output to be NaN (or clamped if comparison does not propagate NaN)
    # For Inf/neginf, it should clamp to the boundaries 5.0 and 0.01 because of:
    # x = torch.where(price <= c_min, 0.01, x) and price >= c_max
    # Inf >= c_max -> 5.0. -Inf <= c_min -> 0.01.
    # NaN comparisons evaluate to False, so x will remain at its last guess or boundary.
    # Let's verify that the outputs are bounded or NaN (not causing segfault or runtime error).

    assert solved[1].item() == 5.0, (
        f"Expected Inf price to clamp to 5.0, got {solved[1].item()}"
    )
    assert solved[2].item() == 0.01, (
        f"Expected -Inf price to clamp to 0.01, got {solved[2].item()}"
    )
    # check that solved is not empty or none
    assert solved.shape == (3,)
