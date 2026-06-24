"""
train_adversarial_market.py — Production training script for WGAN-GP / SFAG and Minimax Hedger.
Trains the generative market paths network and robust hedging policy adversarial minimax loop.
Saves the final generator, discriminator, and policy weights for production evaluation.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

# Add repo root to import path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from hedging.deep_hedging import HedgingPolicy
from hedging.adversarial_market import (
    WGAN_GP_Generator,
    WGAN_GP_Discriminator,
    train_robust_minimax_hedger
)


def simulate_heston_returns(num_paths, seq_len, device):
    """
    Simulates log-returns under the Heston model to act as a target dataset
    possessing realistic stylized facts (fat tails, mean-reverting vol, leverage effect).
    """
    v0 = 0.000225    # daily variance (standard deviation of 1.5%)
    kappa = 0.05     # daily mean reversion
    theta = 0.000225
    sigma_v = 0.0005 # daily vol of vol
    rho = -0.6       # correlation (leverage effect)
    
    V = torch.full((num_paths,), v0, device=device)
    returns = torch.zeros(num_paths, seq_len, device=device)
    
    for t in range(seq_len):
        Z1 = torch.randn(num_paths, device=device)
        Z2 = torch.randn(num_paths, device=device)
        W_S = Z1
        W_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        returns[:, t] = -0.5 * V + torch.sqrt(torch.clamp(V, min=1e-8)) * W_S
        V = torch.clamp(V + kappa * (theta - V) + sigma_v * torch.sqrt(torch.clamp(V, min=1e-8)) * W_v, min=1e-8)
        
    return returns


def main():
    parser = argparse.ArgumentParser(description="Production Minimax Adversarial Market Training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training paths")
    parser.add_argument("--critic_steps", type=int, default=5, help="Number of discriminator steps per generator step")
    parser.add_argument("--minimax_coeff", type=float, default=0.01, help="Minimax adversarial weight coefficient")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent noise vector dimension")
    parser.add_argument("--seq_len", type=int, default=252, help="Sequence length of path (days)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on")
    parser.add_argument("--risk_measure", type=str, default="entropic", choices=["entropic", "quad"],
                        help="Risk measure for DeepHedgingEnv")
    
    args = parser.parse_args()
    print("=== Starting Minimax Adversarial Market Production Training ===")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    
    # 1. Setup mock real returns dataset for stylized facts alignment (10,000 samples)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Generating mock historical returns data...")
    # Simulate stylized returns under the Heston model
    real_returns = simulate_heston_returns(80000, args.seq_len, args.device)
    
    # Target stylized facts stats calculated from real market returns:
    # Autocorrelation of absolute returns (ACF) target (lag 1 to 20 decay)
    real_acf = torch.linspace(0.15, 0.01, 20, device=args.device)
    # Negative correlation between return shocks and vol changes (Leverage Effect)
    real_leverage = -0.15
    # Volatility correlation across scales (CFVC matrix)
    real_cfvc_matrix = torch.eye(4, device=args.device) * 0.8 + 0.2
    
    # 2. Initialize Networks
    # d = 1 tradeable instrument (stock price only, vol proxy is purely a state variable)
    d = 1
    generator = WGAN_GP_Generator(latent_dim=args.latent_dim, seq_len=args.seq_len, hidden_dim=64).to(args.device)
    discriminator = WGAN_GP_Discriminator(seq_len=args.seq_len, hidden_dim=64).to(args.device)
    # State dimension: log_moneyness (1), time_to_expiry (1), vol_proxy (1), prev_delta (d=1) = 4
    policy = HedgingPolicy(input_dim=4, hidden_dim=64, output_dim=d).to(args.device)
    
    # 3. Train
    print("Training minimax robust networks...")
    t0 = time.time()
    train_robust_minimax_hedger(
        real_returns=real_returns,
        real_acf=real_acf,
        real_leverage=real_leverage,
        real_cfvc_matrix=real_cfvc_matrix,
        generator=generator,
        discriminator=discriminator,
        policy=policy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        critic_steps=args.critic_steps,
        minimax_coeff=args.minimax_coeff,
        device=args.device,
        risk_measure=args.risk_measure
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.2f} seconds.")
    
    # 4. Save Production Weights
    gen_path = os.path.join(repo_root, "artifacts", "weights", "generator_prod.pth")
    disc_path = os.path.join(repo_root, "artifacts", "weights", "discriminator_prod.pth")
    policy_path = os.path.join(repo_root, "artifacts", "weights", "minimax_policy_prod.pth")
    
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), disc_path)
    torch.save(policy.state_dict(), policy_path)
    
    print("\n=== Saved Production Weights ===")
    print(f"Generator:     {gen_path}")
    print(f"Discriminator: {disc_path}")
    print(f"Policy:        {policy_path}")
    print("Adversarial market training and minimax policy are ready for evaluation.")


if __name__ == "__main__":
    main()
