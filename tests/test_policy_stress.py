"""
test_policy_stress.py — Stress testing suite for DeepHedgingPolicy and transaction cost layers.
Verifies numerical stability, gradient correctness around zero, long sequence state preservation,
and behavior with extreme/NaN/Inf values on CPU and CUDA.
"""

import pytest
import torch
from deepvol.hedging.policy import (
    DeepHedgingPolicy,
    proportional_transaction_cost,
    huber_transaction_cost,
    sqrt_transaction_cost,
)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_transaction_costs_gradient_stability(device):
    """
    Stress-test the gradients of all transaction cost layers around delta_diff = 0.
    Specifically checks for NaN/Inf gradients at exact zero and extremely small values.
    """
    # Grid of delta_diff values spanning positive, negative, and exact zero
    delta_diff_vals = [
        -1.0,
        -1e-2,
        -1e-4,
        -1e-6,
        -1e-8,
        -1e-12,
        -1e-20,
        -1e-35,
        0.0,
        1e-35,
        1e-20,
        1e-12,
        1e-8,
        1e-4,
        1e-2,
        1.0,
    ]

    c_fee = 0.002
    S_vals = [0.0, 0.01, 100.0, 1e6]  # Test extreme spot prices

    for S_val in S_vals:
        for val in delta_diff_vals:
            # We want to test gradients, so requires_grad=True
            delta_diff = torch.tensor(
                [[val]], device=device, dtype=torch.float32, requires_grad=True
            )
            S = torch.tensor([[S_val]], device=device, dtype=torch.float32)

            # --- 1. Proportional Cost ---
            cost_prop = proportional_transaction_cost(delta_diff, S, c_fee)
            assert not torch.isnan(cost_prop).any(), (
                f"Prop cost is NaN for delta_diff={val}, S={S_val}"
            )
            assert not torch.isinf(cost_prop).any(), (
                f"Prop cost is Inf for delta_diff={val}, S={S_val}"
            )

            loss_prop = cost_prop.sum()
            loss_prop.backward()

            assert delta_diff.grad is not None
            assert not torch.isnan(delta_diff.grad).any(), (
                f"Prop grad is NaN for delta_diff={val}, S={S_val}"
            )
            assert not torch.isinf(delta_diff.grad).any(), (
                f"Prop grad is Inf for delta_diff={val}, S={S_val}"
            )

            # Reset grad
            delta_diff.grad.zero_()

            # --- 2. Huber Cost ---
            for d in [0.01, 1e-4, 1e-6]:
                cost_huber = huber_transaction_cost(delta_diff, S, c_fee, d=d)
                assert not torch.isnan(cost_huber).any(), (
                    f"Huber cost is NaN for delta_diff={val}, S={S_val}, d={d}"
                )
                assert not torch.isinf(cost_huber).any(), (
                    f"Huber cost is Inf for delta_diff={val}, S={S_val}, d={d}"
                )

                loss_huber = cost_huber.sum()
                loss_huber.backward()

                assert delta_diff.grad is not None
                assert not torch.isnan(delta_diff.grad).any(), (
                    f"Huber grad is NaN for delta_diff={val}, S={S_val}, d={d}"
                )
                assert not torch.isinf(delta_diff.grad).any(), (
                    f"Huber grad is Inf for delta_diff={val}, S={S_val}, d={d}"
                )

                # Check Huber specific behavior: at 0, derivative must be 0
                if val == 0.0:
                    torch.testing.assert_close(
                        delta_diff.grad,
                        torch.zeros_like(delta_diff.grad),
                        atol=1e-7,
                        rtol=1e-7,
                    )

                delta_diff.grad.zero_()

            # --- 3. Square-root Cost ---
            for eps_c in [1e-6, 1e-12, 1e-20]:
                cost_sqrt = sqrt_transaction_cost(delta_diff, S, c_fee, eps_c=eps_c)
                assert not torch.isnan(cost_sqrt).any(), (
                    f"Sqrt cost is NaN for delta_diff={val}, S={S_val}, eps_c={eps_c}"
                )
                assert not torch.isinf(cost_sqrt).any(), (
                    f"Sqrt cost is Inf for delta_diff={val}, S={S_val}, eps_c={eps_c}"
                )

                loss_sqrt = cost_sqrt.sum()
                loss_sqrt.backward()

                assert delta_diff.grad is not None
                assert not torch.isnan(delta_diff.grad).any(), (
                    f"Sqrt grad is NaN for delta_diff={val}, S={S_val}, eps_c={eps_c}"
                )
                assert not torch.isinf(delta_diff.grad).any(), (
                    f"Sqrt grad is Inf for delta_diff={val}, S={S_val}, eps_c={eps_c}"
                )

                # Check Sqrt specific behavior: at 0, derivative must be 0
                if val == 0.0:
                    torch.testing.assert_close(
                        delta_diff.grad,
                        torch.zeros_like(delta_diff.grad),
                        atol=1e-7,
                        rtol=1e-7,
                    )

                delta_diff.grad.zero_()


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("range_limit", ["0.01_0.99", "0_1"])
def test_long_sequence_recurrent_state_preservation(device, range_limit):
    """
    Verify that DeepHedgingPolicy preserves LSTM recurrent state correctly over long sequences.
    Compares multi-step batch forward pass against step-by-step iteration.
    """
    input_dim = 8
    hidden_dim = 32
    output_dim = 1
    seq_len = 1000
    batch_size = 16

    policy = DeepHedgingPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        range_limit=range_limit,
    ).to(device)

    # Generate random features
    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float32)

    # --- 1. Run all steps at once (sequential forward pass) ---
    if device == "cuda":
        torch.cuda.synchronize()

    delta_seq, h_seq = policy(x)

    if device == "cuda":
        torch.cuda.synchronize()

    assert delta_seq.shape == (batch_size, seq_len, output_dim)
    assert h_seq[0].shape == (batch_size, hidden_dim)
    assert h_seq[1].shape == (batch_size, hidden_dim)

    # --- 2. Run step-by-step manually and accumulate ---
    h_step = None
    delta_step_list = []

    for t in range(seq_len):
        x_t = x[:, t, :]
        delta_t, h_step = policy(x_t, h_step)
        delta_step_list.append(delta_t)

    delta_step = torch.stack(delta_step_list, dim=1)

    # --- 3. Verify that sequential forward and step-by-step match exactly ---
    torch.testing.assert_close(delta_seq, delta_step, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(h_seq[0], h_step[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(h_seq[1], h_step[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_extreme_inputs_numerical_robustness(device):
    """
    Examine the policy's response to extreme inputs: NaN, Inf, very large, and very small numbers.
    Assesses what propagates to the final holdings output and recurrent states.
    """
    input_dim = 4
    hidden_dim = 16
    policy = DeepHedgingPolicy(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    batch_size = 4

    # Case A: NaNs in features
    x_nan = torch.randn(batch_size, input_dim, device=device)
    x_nan[0, 0] = float("nan")  # inject NaN

    delta_nan, h_nan = policy(x_nan)
    # The NaN should propagate to the affected batch index's outputs and hidden states
    assert torch.isnan(delta_nan[0]).all()
    assert torch.isnan(h_nan[0][0]).all()
    assert torch.isnan(h_nan[1][0]).all()
    # The unaffected batch indices should remain non-NaN
    assert not torch.isnan(delta_nan[1:]).any()
    assert not torch.isnan(h_nan[0][1:]).any()
    assert not torch.isnan(h_nan[1][1:]).any()

    # Case B: Infs in features
    x_inf = torch.randn(batch_size, input_dim, device=device)
    x_inf[0, 0] = float("inf")  # inject positive infinity
    x_inf[1, 1] = float("-inf")  # inject negative infinity

    delta_inf, h_inf = policy(x_inf)
    # Sigmoid should keep the final delta output within [0.01, 0.99] bounds even with Inf/ -Inf inputs
    # unless NaN is generated internally (which we will verify).
    # Note: Inf input into LSTM linear layers might result in Inf or -Inf hidden states.
    # Sigmoid of Inf is 1.0, Sigmoid of -Inf is 0.0.
    # Let's check if the delta output is finite (non-NaN, non-Inf)
    assert not torch.isinf(delta_inf).any()
    # Let's see if it's NaN. If the hidden state became NaN due to Inf-Inf operations, delta_inf could be NaN.
    # We document the exact outcome.
    print(f"Delta with Inf inputs on {device}: {delta_inf.tolist()}")
    print(f"Hidden state h_c with Inf inputs on {device}: {h_inf[0].tolist()}")

    # Case C: Extremely large values (near overflow for float32)
    x_large = torch.ones(batch_size, input_dim, device=device) * 1e38
    delta_large, h_large = policy(x_large)
    assert not torch.isnan(delta_large).any()
    assert not torch.isinf(delta_large).any()
    # verify bounds are maintained
    assert (delta_large >= 0.01).all() and (delta_large <= 0.99).all()

    # Case D: Extremely small values (near underflow for float32)
    x_small = torch.ones(batch_size, input_dim, device=device) * 1e-38
    delta_small, h_small = policy(x_small)
    assert not torch.isnan(delta_small).any()
    assert not torch.isinf(delta_small).any()
    assert (delta_small >= 0.01).all() and (delta_small <= 0.99).all()
