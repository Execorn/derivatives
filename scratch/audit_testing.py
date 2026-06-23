import os
import sys
import torch
import numpy as np

# Ensure src/ is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hedging.deep_hedging import (
    DeepHedgingEnv,
    HedgingPolicy,
    estimate_gpd_tail_index_pwm,
    compute_leverage_loss,
    compute_cfvc_loss
)
from hedging.barrier_hedging import BarrierHedgingEnv

def run_pwm_instability_test():
    print("=== TEST 1: estimate_gpd_tail_index_pwm Instabilities ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Scenario A: n_exceed = 0
    # Let returns be length 100, threshold_quantile = 1.0 (or close to it)
    print("\n--- Scenario A: n_exceed is 0 (threshold_quantile close to 1.0) ---")
    returns = torch.randn(10, 100, device=device)
    try:
        xi = estimate_gpd_tail_index_pwm(returns, threshold_quantile=0.99999)
        print(f"n_exceed=0 output shape: {xi.shape}, values: {xi}")
        if torch.isnan(xi).any() or torch.isinf(xi).any():
            print("WARNING: NaNs or Infs detected in tail index!")
    except Exception as e:
        print(f"Exception raised: {e}")
        
    # Scenario B: Denominator is exactly 0.0
    # For N=2 exceedances, a0 - 2*a1 = 0 when y2 = 1.3076923 * y1
    print("\n--- Scenario B: Denominator is exactly 0.0 ---")
    y1 = 1.0
    y2 = 1.3076923076923077
    # We want these to be the exceedances. Let's construct a tensor.
    # To get these exact exceedances, we can set the returns to have these as the top 2 elements.
    returns_zero_denom = torch.zeros(1, 10, device=device)
    returns_zero_denom[0, -2] = y1
    returns_zero_denom[0, -1] = y2
    # with threshold_quantile = 0.8, n_exceed = 2
    try:
        xi = estimate_gpd_tail_index_pwm(returns_zero_denom, threshold_quantile=0.8)
        print(f"Exactly 0.0 denominator output: {xi.item()}")
    except Exception as e:
        print(f"Exception raised: {e}")
        
    # Scenario C: Denominator is extremely close to 0.0 (near-underflow) in float64/float32
    print("\n--- Scenario C: Denominator is extremely close to 0.0 ---")
    # For N=2, a0 - 2*a1 = -0.425 * y1 + 0.325 * y2
    # If we choose y1 = 1.0, and y2 = 1.3076923 + 1e-7, then denominator is ~3.25e-8
    # If we choose y2 = 1.30769230769 + 1e-9 (in float64), denominator is ~3.25e-10.
    # Let's construct a tensor where the denominator is around 1e-9 (very close to zero)
    y1 = 1.0
    y2 = 1.3076923 + 1e-7 # perturbed slightly in float32 range
    returns_tiny_denom = torch.tensor([[0.0]*8 + [y1, y2]], device=device, dtype=torch.float32)
    try:
        # We temporarily disable the replacement or simulate what happens without clamping
        # Here we just show the output of the function itself
        xi = estimate_gpd_tail_index_pwm(returns_tiny_denom, threshold_quantile=0.8)
        print(f"Perturbed tiny denominator output: {xi.item()}")
        
        # Now let's calculate what happens mathematically if denominator is 1e-15 (near zero but not 0)
        # in the formula: xi = 2.0 - a0 / denominator
        # If a0 = 1.15 and denominator = 1e-15, xi = 2.0 - 1.15e15 = -1.15e15 (overflow!)
        print("Mathematical risk: If denominator is e.g. 1e-15, xi is calculated as:")
        a0_val = 1.15
        denom_val = 1e-15
        xi_val = 2.0 - a0_val / denom_val
        print(f"  xi = {xi_val} (causing massive gradient explosion and NaNs)")
    except Exception as e:
        print(f"Exception raised: {e}")

def run_env_instability_test():
    print("\n=== TEST 2: Environment State Instabilities (Log of zero/neg) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Scenario A: DeepHedgingEnv with non-positive asset prices
    print("\n--- Scenario A: DeepHedgingEnv with non-positive prices ---")
    H_bad = torch.zeros(2, 10, 2, device=device)
    # stock price goes to 0 or negative
    H_bad[0, :, 0] = torch.tensor([100.0, 90.0, 80.0, 50.0, 0.0, -10.0, -20.0, 10.0, 20.0, 30.0])
    H_bad[1, :, 0] = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]) # constant
    
    # Volatility channel
    H_bad[:, :, 1] = 0.2
    
    payoff = torch.zeros(2, device=device)
    cost_coeffs = torch.tensor([0.001, 0.001], device=device)
    
    env = DeepHedgingEnv(H=H_bad, payoff=payoff, cost_coeffs=cost_coeffs, strike=100.0, expiry=1.0)
    
    # Check step 4 (where price is 0.0) and step 5 (where price is -10.0)
    for k in [3, 4, 5]:
        print(f"Step {k} prices: {H_bad[:, k, 0].tolist()}")
        try:
            state = env.get_state(k, torch.zeros(2, 2, device=device))
            print(f"State constructed: {state}")
            if torch.isnan(state).any() or torch.isinf(state).any():
                print("WARNING: NaNs or Infs detected in DeepHedgingEnv state!")
        except Exception as e:
            print(f"Exception raised at step {k}: {e}")
            
    # Scenario B: BarrierHedgingEnv with spot touching/crossing barrier
    print("\n--- Scenario B: BarrierHedgingEnv with spot touching/crossing barrier ---")
    # Barrier is 85.0. Let's make spot hit 85.0 exactly, and go below 85.0
    H_barrier = torch.zeros(3, 5, 2, device=device)
    H_barrier[0, :, 0] = torch.tensor([100.0, 90.0, 85.0, 80.0, 75.0])  # hits and crosses
    H_barrier[1, :, 0] = torch.tensor([100.0, 95.0, 85.00001, 90.0, 95.0]) # extremely close to barrier
    H_barrier[2, :, 0] = torch.tensor([100.0, 90.0, -10.0, 80.0, 90.0]) # negative price
    H_barrier[:, :, 1] = 0.2
    
    env_barrier = BarrierHedgingEnv(H=H_barrier, cost_coeffs=cost_coeffs, strike=100.0, barrier=85.0, expiry=1.0)
    
    active_mask = torch.ones(3, 1, device=device)
    prev_delta = torch.zeros(3, 2, device=device)
    
    for k in range(5):
        print(f"Step {k} prices: {H_barrier[:, k, 0].tolist()}")
        try:
            state = env_barrier.get_state(k, prev_delta, active_mask)
            print(f"State constructed: {state}")
            if torch.isnan(state).any() or torch.isinf(state).any():
                print("WARNING: NaNs or Infs detected in BarrierHedgingEnv state!")
        except Exception as e:
            print(f"Exception raised at step {k}: {e}")

def run_other_losses_test():
    print("\n=== TEST 3: Other Losses (compute_leverage_loss, compute_cfvc_loss) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Constant returns (flat path)
    returns = torch.zeros(5, 50, device=device)
    target_leverage = -0.15
    target_corr = torch.eye(4, device=device)
    
    print("\n--- compute_leverage_loss with flat paths ---")
    try:
        loss = compute_leverage_loss(returns, target_leverage, vol_window=20)
        print(f"Leverage loss with flat paths: {loss.item()}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: Leverage loss is NaN/Inf!")
    except Exception as e:
        print(f"Exception raised: {e}")
        
    print("\n--- compute_cfvc_loss with flat paths ---")
    try:
        loss = compute_cfvc_loss(returns, target_corr, scales=[5, 20, 60])
        print(f"CFVC loss with flat paths: {loss.item()}")
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: CFVC loss is NaN/Inf!")
    except Exception as e:
        print(f"Exception raised: {e}")

def run_barrier_boundary_hedging_leak_test():
    print("\n=== TEST 4: Barrier Knockout Boundary Trading & Leak Test ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 5 paths, 10 steps, d=2
    # Path 0: knocked out early (step 2)
    # Path 1: never knocked out
    # Path 2: knocked out at step 5
    # Path 3: hits barrier exactly at step 3
    # Path 4: knocked out at step 1
    H = torch.zeros(5, 11, 2, device=device)
    H[0, :, 0] = torch.tensor([100.0, 95.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]) # knocked out at k=2
    H[1, :, 0] = torch.tensor([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0]) # never knocked out
    H[2, :, 0] = torch.tensor([100.0, 95.0, 90.0, 92.0, 95.0, 84.0, 80.0, 80.0, 80.0, 80.0, 80.0]) # knocked out at k=5
    H[3, :, 0] = torch.tensor([100.0, 95.0, 90.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0]) # hits 85.0 exactly at k=3
    H[4, :, 0] = torch.tensor([100.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]) # knocked out at k=1
    
    H[:, :, 1] = 0.2  # dummy vol
    
    cost_coeffs = torch.tensor([0.01, 0.05], device=device)  # High cost to see the difference clearly
    
    env = BarrierHedgingEnv(
        H=H,
        cost_coeffs=cost_coeffs,
        strike=100.0,
        barrier=85.0,
        expiry=1.0
    )
    
    # We want a policy that outputs non-zero deltas to see if it keeps trading after knockout
    # Let's initialize a policy and make its weights non-zero
    policy = HedgingPolicy(input_dim=6, hidden_dim=16, output_dim=2).to(device)
    # Set bias of fc to non-zero so output delta is guaranteed non-zero
    for layer in policy.fc:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.constant_(layer.bias, 0.5)
            torch.nn.init.constant_(layer.weight, 0.1)
            
    wealth, total_costs, all_deltas = env.simulate_hedging_episode(policy)
    
    print("\nResults:")
    for i in range(5):
        print(f"Path {i}:")
        print(f"  Spot path: {[round(x, 1) for x in H[i, :, 0].tolist()]}")
        print(f"  Final Payoff: {env.payoff[i].item()}")
        print(f"  Final Wealth: {wealth[i].item():.4f}")
        print(f"  Total Transaction Costs: {total_costs[i].item():.4f}")
        # print first few deltas after knockout
        # Path 0 knocks out at k=2. Let's see deltas for k=0,1,2,3,4
        deltas_path = all_deltas[i].cpu().tolist()
        print(f"  Deltas (k=0 to 9): {[[round(d[0], 3), round(d[1], 3)] for d in deltas_path[:10]]}")

def run_vram_leak_test():
    print("\n=== TEST 5: GPU/VRAM Memory Leak Profiling ===")
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping VRAM leak profiling.")
        return
        
    device = "cuda"
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated(device)
    print(f"Initial allocated memory: {start_mem / 1e6:.3f} MB")
    
    # Simulate a small environment and run policy in a loop
    N_paths = 100
    N_t = 100
    d = 2
    H = torch.randn(N_paths, N_t + 1, d, device=device)
    H[:, :, 0] = H[:, :, 0] * 20.0 + 100.0 # scale spot
    H[:, :, 1] = torch.clamp(torch.nn.functional.softplus(H[:, :, 1]), min=1e-4) # scale vol
    
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    cost_coeffs = torch.tensor([0.001, 0.005], device=device)
    
    env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, strike=100.0, expiry=1.0)
    policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    print("Running 30 iterations of hedging and backprop...")
    for step in range(30):
        optimizer.zero_grad()
        wealth, _, _ = env.simulate_hedging_episode(policy)
        loss = env.compute_loss(wealth)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            mem = torch.cuda.memory_allocated(device)
            print(f"  Iteration {step+1:02d}: Allocated memory = {mem / 1e6:.3f} MB")
            
    torch.cuda.empty_cache()
    end_mem = torch.cuda.memory_allocated(device)
    print(f"Final allocated memory (after empty_cache): {end_mem / 1e6:.3f} MB")
    leak = end_mem - start_mem
    print(f"Memory change: {leak / 1e6:.3f} MB")
    if leak > 5e6: # More than 5MB increase
        print("WARNING: VRAM usage increased significantly! Potential memory leak.")
    else:
        print("VRAM usage is stable. No leak detected.")

if __name__ == "__main__":
    run_pwm_instability_test()
    run_env_instability_test()
    run_other_losses_test()
    run_barrier_boundary_hedging_leak_test()
    run_vram_leak_test()
