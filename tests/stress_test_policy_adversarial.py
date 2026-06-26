"""
stress_test_policy_adversarial.py — Stress tests for DeepHedgingPolicy and transaction cost layers.
"""

import os
import sys
import torch
import numpy as np
import importlib

# Setup project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(project_root, "src"))

import deepvol.hedging.policy as policy_module  # noqa: E402


def log_test_header(name: str):
    print("\n" + "=" * 80)
    print(f" RUNNING STRESS TEST: {name}")
    print("=" * 80)


def run_nan_inf_propagation(device: torch.device):
    log_test_header(f"NaN / Inf Propagation on {device}")

    input_dim = 5
    hidden_dim = 16
    output_dim = 1

    policy = policy_module.DeepHedgingPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        range_limit="0.01_0.99",
    ).to(device)

    batch_size = 4

    # Case 1: Pure NaN input
    x_nan = torch.full(
        (batch_size, input_dim), float("nan"), device=device, dtype=torch.float32
    )
    delta_nan, h_nan = policy(x_nan)
    print("NaN input:")
    print("  delta_nan:", delta_nan)
    print("  h_nan (hidden state):", h_nan[0])
    print("  c_nan (cell state):", h_nan[1])

    assert torch.isnan(delta_nan).all(), "NaN input should produce NaN delta"
    assert torch.isnan(h_nan[0]).all(), "NaN input should produce NaN hidden state"
    assert torch.isnan(h_nan[1]).all(), "NaN input should produce NaN cell state"

    # Case 2: Pure Positive Inf input
    x_inf = torch.full(
        (batch_size, input_dim), float("inf"), device=device, dtype=torch.float32
    )
    delta_inf, h_inf = policy(x_inf)
    print("Positive Inf input:")
    print("  delta_inf:", delta_inf)
    print("  h_inf (hidden state):", h_inf[0])

    # Case 3: Pure Negative Inf input
    x_ninf = torch.full(
        (batch_size, input_dim), float("-inf"), device=device, dtype=torch.float32
    )
    delta_ninf, h_ninf = policy(x_ninf)
    print("Negative Inf input:")
    print("  delta_ninf:", delta_ninf)

    # Case 4: Extreme values
    x_extreme = torch.full(
        (batch_size, input_dim), 1e12, device=device, dtype=torch.float32
    )
    delta_ext, h_ext = policy(x_extreme)
    print("Extreme large input (1e12):")
    print("  delta_ext:", delta_ext)
    assert not torch.isnan(delta_ext).any(), "Extreme inputs should not yield NaN"
    assert torch.all(delta_ext >= 0.01) and torch.all(delta_ext <= 0.99), (
        "Clamping boundary failed for extreme inputs"
    )

    # Case 5: Extremely small values
    x_tiny = torch.full(
        (batch_size, input_dim), 1e-12, device=device, dtype=torch.float32
    )
    delta_tiny, h_tiny = policy(x_tiny)
    print("Extreme tiny input (1e-12):")
    print("  delta_tiny:", delta_tiny)
    assert not torch.isnan(delta_tiny).any(), "Tiny inputs should not yield NaN"
    assert torch.all(delta_tiny >= 0.01) and torch.all(delta_tiny <= 0.99), (
        "Clamping boundary failed for tiny inputs"
    )

    print("NaN/Inf propagation tests completed successfully!")


def run_long_sequence_recurrent_preservation(device: torch.device):
    log_test_header(f"Long Sequence Recurrent State Preservation on {device}")

    input_dim = 10
    hidden_dim = 32
    output_dim = 2
    seq_len = 1000
    batch_size = 8

    policy = policy_module.DeepHedgingPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        range_limit="0.01_0.99",
    ).to(device)

    # Generate random input sequence
    torch.manual_seed(42)
    x_seq = torch.randn(
        batch_size, seq_len, input_dim, device=device, dtype=torch.float32
    )

    # 1. Sequential forward pass (full 3D tensor)
    delta_seq, h_seq_final = policy(x_seq)

    # 2. Step-by-step forward pass (recurrent state preservation)
    h_step = None
    delta_steps = []
    for t in range(seq_len):
        x_t = x_seq[:, t, :]
        delta_t, h_step = policy(x_t, h_step)
        delta_steps.append(delta_t)

    delta_step_tensor = torch.stack(delta_steps, dim=1)

    # Compare outputs
    torch.testing.assert_close(delta_seq, delta_step_tensor, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(h_seq_final[0], h_step[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(h_seq_final[1], h_step[1], rtol=1e-5, atol=1e-5)

    print(
        f"Sequence length {seq_len} forward outputs match step-by-step updates exactly."
    )
    print("Final state norms:")
    print(f"  h_c norm: {torch.norm(h_step[0]).item():.4f}")
    print(f"  c_c norm: {torch.norm(h_step[1]).item():.4f}")

    # Ensure no NaN/Inf after 1000 steps
    assert not torch.isnan(h_step[0]).any(), (
        "Hidden state contains NaNs after long sequence"
    )
    assert not torch.isnan(h_step[1]).any(), (
        "Cell state contains NaNs after long sequence"
    )
    assert not torch.isinf(h_step[0]).any(), (
        "Hidden state contains Infs after long sequence"
    )
    assert not torch.isinf(h_step[1]).any(), (
        "Cell state contains Infs after long sequence"
    )

    print("Long sequence recurrent state preservation tests passed!")


def run_transaction_cost_derivatives(device: torch.device):
    log_test_header(f"Transaction Cost Derivatives Stability on {device}")

    c_fee = 0.002
    S = torch.ones(10, 1, device=device, dtype=torch.float32) * 100.0

    # We want to test gradients at and around 0.0
    perturbations = [
        0.0,
        1e-20,
        -1e-20,
        1e-15,
        -1e-15,
        1e-8,
        -1e-8,
        1e-5,
        -1e-5,
        0.05,
        -0.05,
    ]

    for pert in perturbations:
        delta_diff = torch.full(
            (10, 1), pert, device=device, dtype=torch.float32, requires_grad=True
        )

        # 1. Proportional cost
        cost_prop = policy_module.proportional_transaction_cost(delta_diff, S, c_fee)
        loss_prop = cost_prop.sum()
        loss_prop.backward()

        grad_prop = delta_diff.grad.clone()
        delta_diff.grad.zero_()

        # 2. Huber cost
        cost_huber = policy_module.huber_transaction_cost(delta_diff, S, c_fee, d=0.01)
        loss_huber = cost_huber.sum()
        loss_huber.backward()

        grad_huber = delta_diff.grad.clone()
        delta_diff.grad.zero_()

        # 3. Square-root cost
        cost_sqrt = policy_module.sqrt_transaction_cost(
            delta_diff, S, c_fee, eps_c=1e-6
        )
        loss_sqrt = cost_sqrt.sum()
        loss_sqrt.backward()

        grad_sqrt = delta_diff.grad.clone()
        delta_diff.grad.zero_()

        print(f"Perturbation = {pert:+.2e}:")
        print(
            f"  Prop Cost Grad:  {grad_prop[0].item():+.6f} (NaN: {torch.isnan(grad_prop).any().item()}, Inf: {torch.isinf(grad_prop).any().item()})"
        )
        print(
            f"  Huber Cost Grad: {grad_huber[0].item():+.6f} (NaN: {torch.isnan(grad_huber).any().item()}, Inf: {torch.isinf(grad_huber).any().item()})"
        )
        print(
            f"  Sqrt Cost Grad:  {grad_sqrt[0].item():+.6f} (NaN: {torch.isnan(grad_sqrt).any().item()}, Inf: {torch.isinf(grad_sqrt).any().item()})"
        )

        # Verify no NaNs or Infs
        assert not torch.isnan(grad_prop).any(), (
            f"Prop cost gradient is NaN at pert={pert}"
        )
        assert not torch.isnan(grad_huber).any(), (
            f"Huber cost gradient is NaN at pert={pert}"
        )
        assert not torch.isnan(grad_sqrt).any(), (
            f"Sqrt cost gradient is NaN at pert={pert}"
        )

        assert not torch.isinf(grad_prop).any(), (
            f"Prop cost gradient is Inf at pert={pert}"
        )
        assert not torch.isinf(grad_huber).any(), (
            f"Huber cost gradient is Inf at pert={pert}"
        )
        assert not torch.isinf(grad_sqrt).any(), (
            f"Sqrt cost gradient is Inf at pert={pert}"
        )

        # Mathematical checks:
        if pert == 0.0:
            torch.testing.assert_close(
                grad_huber, torch.zeros_like(grad_huber), rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                grad_sqrt, torch.zeros_like(grad_sqrt), rtol=1e-6, atol=1e-6
            )

        if abs(pert) >= 0.01:
            expected_huber_grad = c_fee * S * np.sign(pert)
            torch.testing.assert_close(
                grad_huber, expected_huber_grad, rtol=1e-5, atol=1e-5
            )

        if pert != 0.0:
            expected_prop_grad = c_fee * S * np.sign(pert)
            torch.testing.assert_close(
                grad_prop, expected_prop_grad, rtol=1e-5, atol=1e-5
            )

    print("Transaction cost derivatives tests passed!")


def run_extreme_inputs_transaction_cost(device: torch.device):
    log_test_header(f"Extreme Inputs for Transaction Cost Layer on {device}")

    # 1. Extreme Spot Price
    S_large = torch.ones(5, 1, device=device, dtype=torch.float32) * 1e8
    S_small = torch.ones(5, 1, device=device, dtype=torch.float32) * 1e-8

    delta_diff = torch.tensor(
        [[0.05], [-0.02], [0.0], [0.1], [-0.1]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    cost_large_S = policy_module.huber_transaction_cost(
        delta_diff, S_large, c_fee=0.002, d=0.01
    )
    cost_large_S.sum().backward()
    grad_large_S = delta_diff.grad.clone()
    delta_diff.grad.zero_()

    cost_small_S = policy_module.huber_transaction_cost(
        delta_diff, S_small, c_fee=0.002, d=0.01
    )
    cost_small_S.sum().backward()
    grad_small_S = delta_diff.grad.clone()
    delta_diff.grad.zero_()

    print("Huber Cost with Large Spot S=1e8:")
    print("  cost:", cost_large_S.detach().cpu().numpy().flatten())
    print("  grad:", grad_large_S.detach().cpu().numpy().flatten())
    assert not torch.isnan(grad_large_S).any()

    print("Huber Cost with Tiny Spot S=1e-8:")
    print("  cost:", cost_small_S.detach().cpu().numpy().flatten())
    print("  grad:", grad_small_S.detach().cpu().numpy().flatten())
    assert not torch.isnan(grad_small_S).any()

    # 2. Extreme c_fee
    c_fee_large = 5.0
    c_fee_zero = 0.0

    cost_large_fee = policy_module.huber_transaction_cost(
        delta_diff, S_large, c_fee=c_fee_large, d=0.01
    )
    cost_large_fee.sum().backward()
    grad_large_fee = delta_diff.grad.clone()
    delta_diff.grad.zero_()

    cost_zero_fee = policy_module.huber_transaction_cost(
        delta_diff, S_large, c_fee=c_fee_zero, d=0.01
    )
    cost_zero_fee.sum().backward()
    grad_zero_fee = delta_diff.grad.clone()
    delta_diff.grad.zero_()

    print("Huber Cost with c_fee = 5.0:")
    print("  grad:", grad_large_fee.detach().cpu().numpy().flatten())
    assert not torch.isnan(grad_large_fee).any()

    print("Huber Cost with c_fee = 0.0:")
    print("  grad:", grad_zero_fee.detach().cpu().numpy().flatten())
    torch.testing.assert_close(grad_zero_fee, torch.zeros_like(grad_zero_fee))

    print("Extreme inputs transaction cost tests passed!")


def test_all_adversarial_cases():
    """Pytest entry point."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        run_nan_inf_propagation(device)
        run_long_sequence_recurrent_preservation(device)
        run_transaction_cost_derivatives(device)
        run_extreme_inputs_transaction_cost(device)


def apply_compile_patch():
    print("\nApplying torch.compile patch to map reduce-overhead to default...")
    if not hasattr(torch, "__original_compile"):
        torch.__original_compile = torch.compile

        def mock_compile(model=None, *args, **kwargs):
            if kwargs.get("mode") == "reduce-overhead":
                kwargs["mode"] = "default"
                kwargs["dynamic"] = True
            if model is None:
                return lambda f: mock_compile(f, *args, **kwargs)
            return torch.__original_compile(model, *args, **kwargs)

        torch.compile = mock_compile

    # Reload policy module so that decorators execute with the patched compiler
    global policy_module
    importlib.reload(policy_module)
    print("Module deepvol.hedging.policy reloaded successfully!")


if __name__ == "__main__":
    print("Starting Deep Hedging Policy Adversarial Stress Tests...")

    # Run CPU tests first (always safe)
    run_nan_inf_propagation(torch.device("cpu"))
    run_long_sequence_recurrent_preservation(torch.device("cpu"))
    run_transaction_cost_derivatives(torch.device("cpu"))
    run_extreme_inputs_transaction_cost(torch.device("cpu"))

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")

        # Verify CUDAGraphs Overwrite Vulnerability
        print("\n" + "=" * 80)
        print(" VERIFYING CUDAGRAPHS OVERWRITE VULNERABILITY (UNPATCHED)")
        print("=" * 80)
        try:
            run_transaction_cost_derivatives(cuda_device)
            print(
                "WARNING: run_transaction_cost_derivatives passed on CUDA without the patch! This shouldn't happen unless conftest.py is loaded."
            )
        except RuntimeError as e:
            print("\nSUCCESSFULLY REPRODUCED THE VULNERABILITY!")
            print("Caught expected CUDAGraphs RuntimeError:")
            print(f"  {e}")
            print(
                "\nExplanation: Under torch.compile(mode='reduce-overhead') with CUDAGraphs enabled, backpropagating gradients through multiple compiled transaction cost function invocations overwrites the underlying static CUDAGraph memory buffers, causing a RuntimeError."
            )

        # Apply the patch and verify it works
        apply_compile_patch()

        print("\n" + "=" * 80)
        print(" VERIFYING RUN WITH THE PATCH (EXPECTED TO PASS)")
        print("=" * 80)
        run_nan_inf_propagation(cuda_device)
        run_long_sequence_recurrent_preservation(cuda_device)
        run_transaction_cost_derivatives(cuda_device)
        run_extreme_inputs_transaction_cost(cuda_device)

    print("\nALL ADVERSARIAL STRESS TESTS COMPLETED!")
