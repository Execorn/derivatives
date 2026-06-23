import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure src is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hedging.deep_hedging import HedgingPolicy, DeepHedgingEnv
from src.hedging.barrier_hedging import BarrierHedgingEnv
from src.hedging.adversarial_market import (
    WGAN_GP_Generator,
    WGAN_GP_Discriminator,
    convert_returns_to_prices,
    StylizedFactsAlignmentGAN
)

def simulate_gbm_paths(S0, mu, sigma, T, steps, N_paths, d=2, device="cuda"):
    dt = T / steps
    t_grid = torch.arange(steps + 1, device=device) * dt
    W = torch.randn(N_paths, steps, device=device)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    S = S0 * torch.exp(torch.cumsum(log_returns, dim=-1))
    S0_col = torch.full((N_paths, 1), S0, device=device)
    S_full = torch.cat([S0_col, S], dim=-1)
    if d == 1:
        H = S_full.unsqueeze(-1)
    else:
        vol = torch.full_like(S_full, sigma)
        H = torch.stack([S_full, vol], dim=-1)
    return H, t_grid

def benchmark_env(env_class, batch_size, steps, d, device="cuda"):
    H, t_grid = simulate_gbm_paths(S0=100.0, mu=0.0, sigma=0.2, T=1.0, steps=steps, N_paths=batch_size, d=d, device=device)
    cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)[:d]
    S_T = H[:, -1, 0]
    payoff = torch.clamp(S_T - 100.0, min=0.0)
    
    if env_class == DeepHedgingEnv:
        env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, strike=100.0, expiry=1.0, risk_aversion=1.0, risk_measure="entropic", t_grid=t_grid)
        input_dim = 3 + d
    else:
        env = BarrierHedgingEnv(H=H, cost_coeffs=cost_coeffs, strike=100.0, barrier=85.0, expiry=1.0, risk_aversion=1.0, risk_measure="entropic", t_grid=t_grid)
        input_dim = 4 + d

    policy = HedgingPolicy(input_dim=input_dim, hidden_dim=64, output_dim=d).to(device)
    policy.eval()

    # Warmup
    with torch.no_grad():
        env.precompute = False
        env.simulate_hedging_episode(policy)
        env.precompute = True
        env.simulate_hedging_episode(policy)

    num_runs = 30
    
    # Benchmark unoptimized
    env.precompute = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            env.simulate_hedging_episode(policy)
    torch.cuda.synchronize()
    t_unopt = (time.perf_counter() - t0) / num_runs * 1000.0

    # Benchmark optimized
    env.precompute = True
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            env.simulate_hedging_episode(policy)
    torch.cuda.synchronize()
    t_opt = (time.perf_counter() - t0) / num_runs * 1000.0

    speedup = (t_unopt - t_opt) / t_unopt * 100.0
    return t_unopt, t_opt, speedup

def benchmark_gan_update(device="cuda"):
    batch_size = 256
    seq_len = 252
    latent_dim = 100
    generator = WGAN_GP_Generator(latent_dim=latent_dim, seq_len=seq_len, hidden_dim=128).to(device)
    discriminator = WGAN_GP_Discriminator(seq_len=seq_len, hidden_dim=64).to(device)
    policy = HedgingPolicy(input_dim=5, hidden_dim=64, output_dim=2).to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=1e-4)
    sfag = StylizedFactsAlignmentGAN(generator, discriminator, latent_dim=latent_dim)
    
    real_returns = torch.randn(batch_size, seq_len, device=device) * 0.01
    real_acf = torch.linspace(0.1, 0.0, 20, device=device)
    real_leverage = -0.12
    real_cfvc_matrix = torch.eye(4, device=device)
    cost_coeffs = torch.tensor([0.0001, 0.0005], device=device)

    # Warmup
    opt_g.zero_grad()
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_samples = generator(z)
    fake_ret = fake_samples[:, 0, :]
    fake_vol = fake_samples[:, 1, :]
    d_fake = discriminator(fake_samples)
    g_loss_adv = -torch.mean(d_fake)
    l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(fake_ret, real_returns, real_acf, real_leverage, real_cfvc_matrix)
    g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
    H = convert_returns_to_prices(fake_ret, fake_vol)
    payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
    env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0)
    wealth, _, _ = env.simulate_hedging_episode(policy)
    hedge_loss = env.compute_loss(wealth)
    g_total_loss = g_loss_adv + g_loss_sf - 0.05 * hedge_loss
    g_total_loss.backward()
    opt_g.step()

    num_runs = 20

    # Benchmark with policy requirements_grad frozen (simulated)
    # Since the codebase's train_robust_minimax_hedger is imported, we can just run the generator training step logic here
    # 1. Unfrozen policy parameters (unoptimized)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        opt_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(z)
        fake_ret = fake_samples[:, 0, :]
        fake_vol = fake_samples[:, 1, :]
        d_fake = discriminator(fake_samples)
        g_loss_adv = -torch.mean(d_fake)
        l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(fake_ret, real_returns, real_acf, real_leverage, real_cfvc_matrix)
        g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
        H = convert_returns_to_prices(fake_ret, fake_vol)
        payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
        env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0)
        
        # Policy is unfrozen
        for p in policy.parameters():
            p.requires_grad = True
        wealth, _, _ = env.simulate_hedging_episode(policy)
        hedge_loss = env.compute_loss(wealth)
        g_total_loss = g_loss_adv + g_loss_sf - 0.05 * hedge_loss
        g_total_loss.backward()
        opt_g.step()
    torch.cuda.synchronize()
    t_unfrozen = (time.perf_counter() - t0) / num_runs * 1000.0

    # 2. Frozen policy parameters (optimized)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_runs):
        opt_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_samples = generator(z)
        fake_ret = fake_samples[:, 0, :]
        fake_vol = fake_samples[:, 1, :]
        d_fake = discriminator(fake_samples)
        g_loss_adv = -torch.mean(d_fake)
        l_gpd, l_acf, l_lev, l_cfvc = sfag.compute_stylized_fact_losses(fake_ret, real_returns, real_acf, real_leverage, real_cfvc_matrix)
        g_loss_sf = sfag.w_gpd * l_gpd + sfag.w_acf * l_acf + sfag.w_lev * l_lev + sfag.w_cfvc * l_cfvc
        H = convert_returns_to_prices(fake_ret, fake_vol)
        payoff = torch.clamp(H[:, -1, 0] - 100.0, min=0.0)
        env = DeepHedgingEnv(H=H, payoff=payoff, cost_coeffs=cost_coeffs, risk_aversion=1.0)
        
        # Policy is frozen
        for p in policy.parameters():
            p.requires_grad = False
        wealth, _, _ = env.simulate_hedging_episode(policy)
        hedge_loss = env.compute_loss(wealth)
        g_total_loss = g_loss_adv + g_loss_sf - 0.05 * hedge_loss
        g_total_loss.backward()
        opt_g.step()
        
        # Restore for downstream
        for p in policy.parameters():
            p.requires_grad = True
    torch.cuda.synchronize()
    t_frozen = (time.perf_counter() - t0) / num_runs * 1000.0

    speedup = (t_unfrozen - t_frozen) / t_unfrozen * 100.0
    return t_unfrozen, t_frozen, speedup

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print("\n--- Benchmarking Deep Hedging Rollout (steps=252, d=2) ---")
    for bs in [256, 1024, 4096]:
        t_unopt, t_opt, speedup = benchmark_env(DeepHedgingEnv, bs, 252, 2, device)
        print(f"Batch size: {bs:4d} | Unoptimized: {t_unopt:7.2f} ms | Optimized: {t_opt:7.2f} ms | Time Reduction: {speedup:5.1f}%")

    print("\n--- Benchmarking Barrier Hedging Rollout (steps=252, d=2) ---")
    for bs in [256, 1024, 4096]:
        t_unopt, t_opt, speedup = benchmark_env(BarrierHedgingEnv, bs, 252, 2, device)
        print(f"Batch size: {bs:4d} | Unoptimized: {t_unopt:7.2f} ms | Optimized: {t_opt:7.2f} ms | Time Reduction: {speedup:5.1f}%")

    print("\n--- Benchmarking Generator Update (minimax_coeff=0.05) ---")
    t_unfrozen, t_frozen, speedup = benchmark_gan_update(device)
    print(f"Policy Unfrozen: {t_unfrozen:7.2f} ms | Policy Frozen: {t_frozen:7.2f} ms | Time Reduction: {speedup:5.1f}%")
