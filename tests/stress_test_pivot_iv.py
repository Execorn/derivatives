import pytest
import torch
import math
import sys
import os

# Add src to python path to import deepvol
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from deepvol.hedging.pivot_iv import pivot_implied_vol, price_bs_f64


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_extreme_sk_ratios(device):
    """
    Test spot-strike ratios of 0.01 (deep OTM/ITM depending on option type) and 100.
    At high volatilities (e.g., 2.0), the prices do not underflow, and we verify recovery.
    At low volatilities, the prices underflow/clamped, and we verify clamping to 0.01.
    """
    S = torch.tensor([1.0, 100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0, 1.0], device=device, dtype=torch.float64)
    T = torch.tensor([1.0, 1.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    is_call = torch.tensor([True, True], device=device, dtype=torch.bool)

    # 1. Recoverable case (vol = 2.0)
    true_vol = torch.tensor([2.0, 2.0], device=device, dtype=torch.float64)
    ref_prices = price_bs_f64(true_vol, S, K, T, r, q, is_call)

    for dtype in [torch.float32, torch.float64]:
        ref_p = ref_prices.to(dtype=dtype).clone().requires_grad_(True)
        S_p = S.to(dtype=dtype)
        K_p = K.to(dtype=dtype)
        T_p = T.to(dtype=dtype)
        is_call_p = is_call

        solved_vol = pivot_implied_vol(ref_p, S_p, K_p, T_p, is_call=is_call_p)
        assert torch.allclose(
            solved_vol, torch.tensor([2.0, 2.0], device=device, dtype=dtype), atol=1e-3
        )

        # Check gradient flow
        loss = solved_vol.sum()
        loss.backward()
        assert ref_p.grad is not None
        assert torch.isfinite(ref_p.grad).all()
        assert (ref_p.grad >= 0).all()

    # 2. Clamped case (true vol = 0.2 -> price underflows in double/single precision for extreme moneyness)
    true_vol_low = torch.tensor([0.2, 0.2], device=device, dtype=torch.float64)
    ref_prices_low = price_bs_f64(true_vol_low, S, K, T, r, q, is_call)

    for dtype in [torch.float32, torch.float64]:
        ref_p = ref_prices_low.to(dtype=dtype).clone().requires_grad_(True)
        S_p = S.to(dtype=dtype)
        K_p = K.to(dtype=dtype)
        T_p = T.to(dtype=dtype)

        solved_vol = pivot_implied_vol(ref_p, S_p, K_p, T_p, is_call=is_call)
        # S/K=0.01 underflows to 0.0, S/K=100 underflows to intrinsic S-K = 99.0
        # Both prices are below/equal to c_min (vol=0.01), so they clamp to 0.01
        assert torch.allclose(
            solved_vol,
            torch.tensor([0.01, 0.01], device=device, dtype=dtype),
            atol=1e-6,
        )

        loss = solved_vol.sum()
        loss.backward()
        assert ref_p.grad is not None
        assert torch.isfinite(ref_p.grad).all()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_extreme_maturities(device):
    """
    Test maturity T extremely small (1e-6) and extremely large (10.0).
    """
    S = torch.tensor([100.0, 100.0], device=device, dtype=torch.float64)
    K = torch.tensor([100.0, 100.0], device=device, dtype=torch.float64)
    T = torch.tensor([1e-6, 10.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0], device=device, dtype=torch.float64)
    is_call = torch.tensor([True, True], device=device, dtype=torch.bool)

    # Verify volatility recovery at vol = 0.2
    true_vol = torch.tensor([0.2, 0.2], device=device, dtype=torch.float64)
    ref_prices = price_bs_f64(true_vol, S, K, T, r, q, is_call)

    for dtype in [torch.float32, torch.float64]:
        ref_p = ref_prices.to(dtype=dtype).clone().requires_grad_(True)
        S_p = S.to(dtype=dtype)
        K_p = K.to(dtype=dtype)
        T_p = T.to(dtype=dtype)

        solved_vol = pivot_implied_vol(ref_p, S_p, K_p, T_p, is_call=is_call)
        assert torch.allclose(
            solved_vol, torch.tensor([0.2, 0.2], device=device, dtype=dtype), atol=1e-3
        )

        # Check gradient flow
        loss = solved_vol.sum()
        loss.backward()
        assert ref_p.grad is not None
        assert torch.isfinite(ref_p.grad).all()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_clamping_boundaries(device):
    """
    Test clamping to 0.01 (100 bps) and 5.0 boundaries under:
    - Zero price or negative price
    - Price close to intrinsic
    - Price close to S (extremely high)
    - Infinities or large prices
    """
    S = torch.tensor(
        [100.0, 100.0, 100.0, 100.0, 100.0], device=device, dtype=torch.float64
    )
    K = torch.tensor(
        [100.0, 100.0, 100.0, 100.0, 100.0], device=device, dtype=torch.float64
    )
    T = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float64)
    r = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    is_call = torch.tensor(
        [True, True, True, True, True], device=device, dtype=torch.bool
    )

    # Price scenarios:
    # 1. 0.0 (below c_min, since c_min is call price at vol=0.01)
    # 2. -5.0 (negative)
    # 3. 0.0001 (below c_min, which is ~0.398)
    # 4. 99.9 (close to S=100.0, above c_max which is ~98.0)
    # 5. 150.0 (extremely large, above S=100.0)
    prices = torch.tensor(
        [0.0, -5.0, 0.0001, 99.9, 150.0], device=device, dtype=torch.float64
    )

    for dtype in [torch.float32, torch.float64]:
        p_tensor = prices.to(dtype=dtype).clone().requires_grad_(True)
        solved_vol = pivot_implied_vol(
            p_tensor,
            S.to(dtype=dtype),
            K.to(dtype=dtype),
            T.to(dtype=dtype),
            r.to(dtype=dtype),
            q.to(dtype=dtype),
            is_call=is_call,
        )

        # Expected clamped values:
        # below/equal c_min -> 0.01
        # above/equal c_max -> 5.0
        expected_vol = torch.tensor(
            [0.01, 0.01, 0.01, 5.0, 5.0], device=device, dtype=dtype
        )
        assert torch.allclose(solved_vol, expected_vol, atol=1e-6)

        # Gradients check
        loss = solved_vol.sum()
        loss.backward()
        assert p_tensor.grad is not None
        assert torch.isfinite(p_tensor.grad).all()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_gradients_low_vega_gating(device):
    """
    Verify that PIVOT low-vega gradient gating prevents gradient explosion in OTM/ITM/boundary regions.
    """
    S = torch.tensor([100.0], device=device, dtype=torch.float64)
    K = torch.tensor([300.0], device=device, dtype=torch.float64)  # Deep OTM Call
    T = torch.tensor([0.05], device=device, dtype=torch.float64)
    r = torch.tensor([0.0], device=device, dtype=torch.float64)
    q = torch.tensor([0.0], device=device, dtype=torch.float64)

    # 0.0 price will result in vol clamping to 0.01, giving vega of 0.0
    p = torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True)

    # Use different vega_epsilons and check the gradient value
    for vega_eps in [1e-3, 1e-4, 1e-5]:
        p.grad = None
        solved_vol = pivot_implied_vol(
            p, S, K, T, r, q, is_call=True, vega_epsilon=vega_eps
        )
        solved_vol.backward()

        # The backward pass does: grad_price = grad_output / gated_vega
        # since vega is near-zero (approx 0), gated_vega clamps to vega_eps,
        # so grad_price should be 1.0 / vega_eps
        expected_grad = 1.0 / vega_eps
        assert math.isclose(p.grad.item(), expected_grad, rel_tol=1e-4)


if __name__ == "__main__":
    # If run as a script, run the tests manually
    print("Running extreme S/K ratio tests...")
    test_extreme_sk_ratios("cpu")
    if torch.cuda.is_available():
        test_extreme_sk_ratios("cuda")

    print("Running extreme maturities tests...")
    test_extreme_maturities("cpu")
    if torch.cuda.is_available():
        test_extreme_maturities("cuda")

    print("Running clamping boundaries tests...")
    test_clamping_boundaries("cpu")
    if torch.cuda.is_available():
        test_clamping_boundaries("cuda")

    print("Running low vega gradient gating tests...")
    test_gradients_low_vega_gating("cpu")
    if torch.cuda.is_available():
        test_gradients_low_vega_gating("cuda")

    print("All tests passed manually!")
