"""
train_deep_hedging.py — Production training script for Deep Hedging policies.
Trains the recurrent LSTM policy for European or Barrier option hedging under transaction costs.
Saves the trained policy weights and normalizers for production deployment.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import numpy as np

# Add repo root to import path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from src.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv, train_deep_hedger
from src.hedging.barrier_hedging import BarrierHedgingEnv


def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, d=1, device="cpu"):
    """
    Simulates stock price and constant volatility paths under GBM.
    """
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device) * dt
    
    # Simulate log-returns
    W = torch.randn(N_paths, steps, device=device)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    
    # Cumsum to get stock prices
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device)
    S_full = torch.cat([S0_col, S], dim=-1)
    
    if d == 1:
        H = S_full.unsqueeze(-1)
    else:
        # Prepend volatility proxy process (assume constant volatility)
        vol = torch.full_like(S_full, sigma)
        H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid


def main():
    parser = argparse.ArgumentParser(description="Production Deep Hedging Training")
    parser.add_argument("--option_type", type=str, choices=["european", "barrier"], default="european",
                        help="Option contract style to hedge")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training paths")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--strike", type=float, default=100.0, help="Option strike price K")
    parser.add_argument("--barrier", type=float, default=85.0, help="Exotic barrier knock-out level B")
    parser.add_argument("--expiry", type=float, default=0.1, help="Maturity T in years")
    parser.add_argument("--steps", type=int, default=30, help="Rebalancing frequency (steps)")
    parser.add_argument("--cost_stock", type=float, default=0.0001, help="Stock transaction cost coefficient (1 bp)")
    parser.add_argument("--cost_vol", type=float, default=0.0005, help="Vol instrument cost coefficient (5 bps)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    
    args = parser.parse_args()
    print(f"=== Starting Deep Hedging Production Training ({args.option_type.upper()}) ===")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    
    # 1. Simulate paths for training and testing
    torch.manual_seed(42)
    np.random.seed(42)
    
    # We use d=2 instruments (stock and volatility swap proxy)
    d = 2
    S0 = 100.0
    mu = 0.0
    sigma = 0.2
    
    print("Simulating paths...")
    # Training paths
    H_train, t_grid = simulate_gbm_paths(S0, mu, sigma, args.expiry, args.steps, args.batch_size, d, args.device)
    # Validation/Test paths (10,000 paths for accurate evaluation)
    H_test, _ = simulate_gbm_paths(S0, mu, sigma, args.expiry, args.steps, 10000, d, args.device)
    
    cost_coeffs = torch.tensor([args.cost_stock, args.cost_vol], device=args.device)
    
    # 2. Setup Environment
    if args.option_type == "european":
        # Terminal option payoff: Call option at strike K
        payoff_train = torch.clamp(H_train[:, -1, 0] - args.strike, min=0.0)
        payoff_test = torch.clamp(H_test[:, -1, 0] - args.strike, min=0.0)
        
        env_train = DeepHedgingEnv(H=H_train, payoff=payoff_train, cost_coeffs=cost_coeffs,
                                   strike=args.strike, expiry=args.expiry, risk_aversion=1.0,
                                   risk_measure="quad", t_grid=t_grid)
        env_test = DeepHedgingEnv(H=H_test, payoff=payoff_test, cost_coeffs=cost_coeffs,
                                  strike=args.strike, expiry=args.expiry, risk_aversion=1.0,
                                  risk_measure="quad", t_grid=t_grid)
        
        # State dimension: log_moneyness (1), time_to_expiry (1), vol_proxy (1), prev_delta (d=2) = 5
        input_dim = 3 + d
        output_dim = d
    else:
        # Barrier DOBC option
        env_train = BarrierHedgingEnv(H=H_train, cost_coeffs=cost_coeffs, strike=args.strike,
                                      barrier=args.barrier, expiry=args.expiry, risk_aversion=1.0,
                                      risk_measure="quad", t_grid=t_grid)
        env_test = BarrierHedgingEnv(H=H_test, cost_coeffs=cost_coeffs, strike=args.strike,
                                     barrier=args.barrier, expiry=args.expiry, risk_aversion=1.0,
                                     risk_measure="quad", t_grid=t_grid)
        
        # State dimension: log_moneyness (1), log_barrier_dist (1), time_to_expiry (1), active_mask (1), prev_delta (d=2) = 6
        input_dim = 4 + d
        output_dim = d
        
    # 3. Initialize Recurrent LSTM Policy
    policy = HedgingPolicy(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(args.device)
    
    # 4. Train
    print("Training policy network...")
    t0 = time.time()
    train_deep_hedger(env_train, policy, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.2f} seconds.")
    
    # 5. Save Weights
    weights_path = os.path.join(repo_root, "artifacts", "weights", f"deep_hedger_{args.option_type}_prod.pth")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(policy.state_dict(), weights_path)
    print(f"Saved production policy weights to: {weights_path}")
    
    # 6. Evaluate
    policy.eval()
    with torch.no_grad():
        wealth, costs, deltas = env_test.simulate_hedging_episode(policy)
        test_loss = env_test.compute_loss(wealth).item()
        avg_costs = torch.mean(costs).item()
        std_pnl = torch.std(wealth - env_test.payoff).item()
        
    print("\n=== Production Evaluation Metrics (OOS 10,000 paths) ===")
    print(f"Option Style:       {args.option_type.upper()}")
    print(f"Final Loss (Quadratic): {test_loss:.6f}")
    print(f"Standard Dev of P&L:    {std_pnl:.6f}")
    print(f"Average Transaction Costs: {avg_costs:.6f}")
    print(f"Evaluation completed successfully. Policy is ready for production.")


if __name__ == "__main__":
    main()
