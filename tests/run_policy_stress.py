"""
run_policy_stress.py — Runs detailed stress testing on DeepHedgingPolicy and transaction cost layers.
Saves detailed log outputs to the challenger's folder.
"""

import sys
import os
import torch

# Ensure we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from deepvol.hedging.policy import (  # noqa: E402
    DeepHedgingPolicy,
    proportional_transaction_cost,
    huber_transaction_cost,
    sqrt_transaction_cost,
)


def run_verification():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_lines = []
    log_lines.append("=== Deep Hedging Policy Stress Test ===")
    log_lines.append(f"Device: {device}")
    log_lines.append(f"PyTorch Version: {torch.__version__}\n")

    # ----------------------------------------------------
    # 1. Transaction Cost Derivatives Verification
    # ----------------------------------------------------
    log_lines.append("--- 1. Transaction Cost Derivatives Verification ---")
    delta_diff_vals = [
        -1.0,
        -1e-2,
        -1e-4,
        -1e-8,
        -1e-12,
        -1e-20,
        -0.0,
        0.0,
        1e-20,
        1e-12,
        1e-8,
        1e-4,
        1e-2,
        1.0,
    ]
    c_fee = 0.002
    S = torch.tensor([[100.0]], device=device, dtype=torch.float32)

    # Proportional
    log_lines.append("Proportional Transaction Cost:")
    log_lines.append(f"{'delta_diff':<12} | {'cost':<12} | {'gradient':<12}")
    log_lines.append("-" * 42)
    for val in delta_diff_vals:
        delta_diff = torch.tensor(
            [[val]], device=device, dtype=torch.float32, requires_grad=True
        )
        cost = proportional_transaction_cost(delta_diff, S, c_fee)
        cost.backward()
        log_lines.append(
            f"{val:<12.2e} | {cost.item():<12.6f} | {delta_diff.grad.item():<12.6f}"
        )

    # Huber
    d = 0.01
    log_lines.append("\nHuber Transaction Cost (d = 0.01):")
    log_lines.append(f"{'delta_diff':<12} | {'cost':<12} | {'gradient':<12}")
    log_lines.append("-" * 42)
    for val in delta_diff_vals:
        delta_diff = torch.tensor(
            [[val]], device=device, dtype=torch.float32, requires_grad=True
        )
        cost = huber_transaction_cost(delta_diff, S, c_fee, d=d)
        cost.backward()
        log_lines.append(
            f"{val:<12.2e} | {cost.item():<12.6f} | {delta_diff.grad.item():<12.6f}"
        )

    # Sqrt
    eps_c = 1e-6
    log_lines.append(f"\nSquare-Root Transaction Cost (eps_c = {eps_c}):")
    log_lines.append(f"{'delta_diff':<12} | {'cost':<12} | {'gradient':<12}")
    log_lines.append("-" * 42)
    for val in delta_diff_vals:
        delta_diff = torch.tensor(
            [[val]], device=device, dtype=torch.float32, requires_grad=True
        )
        cost = sqrt_transaction_cost(delta_diff, S, c_fee, eps_c=eps_c)
        cost.backward()
        log_lines.append(
            f"{val:<12.2e} | {cost.item():<12.6f} | {delta_diff.grad.item():<12.6f}"
        )

    # ----------------------------------------------------
    # 2. Recurrent State and Sequence Length Verification
    # ----------------------------------------------------
    log_lines.append(
        "\n--- 2. LSTM Sequence Scaling & Recurrent State Preservation ---"
    )
    input_dim = 8
    hidden_dim = 32
    output_dim = 1
    seq_len = 1000
    batch_size = 16

    policy = DeepHedgingPolicy(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    ).to(device)

    x = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=torch.float32)

    # Sequential forward
    delta_seq, h_seq = policy(x)

    # Step-by-step
    h_step = None
    delta_step_list = []
    for t in range(seq_len):
        delta_t, h_step = policy(x[:, t, :], h_step)
        delta_step_list.append(delta_t)
    delta_step = torch.stack(delta_step_list, dim=1)

    # Check max difference
    max_diff_delta = torch.max(torch.abs(delta_seq - delta_step)).item()
    max_diff_hc = torch.max(torch.abs(h_seq[0] - h_step[0])).item()
    max_diff_cc = torch.max(torch.abs(h_seq[1] - h_step[1])).item()

    log_lines.append(f"Tested sequence length: {seq_len}")
    log_lines.append(f"Batch size: {batch_size}")
    log_lines.append("Max difference between Sequential vs Step-by-Step:")
    log_lines.append(f"  Delta output:  {max_diff_delta:.6e}")
    log_lines.append(f"  LSTM h state:  {max_diff_hc:.6e}")
    log_lines.append(f"  LSTM c state:  {max_diff_cc:.6e}")

    # ----------------------------------------------------
    # 3. Extreme Inputs Verification
    # ----------------------------------------------------
    log_lines.append("\n--- 3. Extreme Input Feature Propagation ---")

    # Case A: NaN
    x_nan = torch.randn(2, input_dim, device=device)
    x_nan[0, 0] = float("nan")
    delta_nan, h_nan = policy(x_nan)
    log_lines.append("NaN input at index 0:")
    log_lines.append(f"  delta_nan[0] (should be nan): {delta_nan[0].tolist()}")
    log_lines.append(f"  delta_nan[1] (should not be nan): {delta_nan[1].tolist()}")

    # Case B: Inf
    x_inf = torch.randn(2, input_dim, device=device)
    x_inf[0, 0] = float("inf")
    x_inf[1, 0] = float("-inf")
    delta_inf, h_inf = policy(x_inf)
    log_lines.append("Inf inputs at index 0 (+inf) and index 1 (-inf):")
    log_lines.append(f"  delta_inf[0]: {delta_inf[0].tolist()}")
    log_lines.append(f"  delta_inf[1]: {delta_inf[1].tolist()}")
    log_lines.append(f"  h_inf_c[0][:4]: {h_inf[0][0][:4].tolist()}")
    log_lines.append(f"  h_inf_c[1][:4]: {h_inf[0][1][:4].tolist()}")

    log_content = "\n".join(log_lines)
    print(log_content)

    # Save log to challenger folder
    log_path = "/home/execorn/programming/derivatives/.agents/teamwork_preview_challenger_m3_1/stress_test_log.txt"
    with open(log_path, "w") as f:
        f.write(log_content)
    print(f"\nSaved stress test results to {log_path}")


if __name__ == "__main__":
    run_verification()
